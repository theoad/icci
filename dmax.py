from typing import Tuple
import inspect
import lovely_tensors as lt
from PIL.Image import Image, fromarray
from jsonargparse import CLI

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch.distributed
import torchvision.datasets
import torchvision.transforms as T

from torch.nn import Sequential
from torchmetrics.image import PeakSignalNoiseRatio
from datasets import load_dataset

from models import *
from gaussian_ot import *
from train_script import *
from fid import *
from cnn import AutoEncoder as CNNAE
from multi_staged import MultiStageVGGAE, AutoEncoder as AEVGG

lt.monkey_patch()


class LatentDmax:
    """
    Approximate the Dmax extimator (i.e. a "visually pleasing" MMSE) with latent W2 optimal transport
    """
    # @register_to_config
    def __init__(
            self,
            model_name: Optional[Union[ModelName, Literal["stable-diffusion", "ms-vgg"]]] = None,
            sample_size: Optional[int] = 224,  # ignored when `pretrained == True`
            loss: str = "mse_loss",
            embed: Literal["pixel", "channel", "image"] = "pixel",
    ) -> None:
        """
        Initialize the Auto-Encoder and the Optimal Transport model.

        :param model_name: The name of the model from torchvision.models to use as encoder.
        :param sample_size: The resize resolution of images (relevant only if model_name is left unfilled)
        :param loss: The loss function from torch.nn.functional to use for training the autoencoder
        :param embed: The dimension on which the transport is performed:
        ``embed="image"``: transport the whole latent image
        ``embed="pixel"``: transport each needle separately
        ``embed="channel"``: transport each channel separately
        """
        super().__init__()
        self.model_name = model_name.lower() if model_name is not None else "vanilla"
        self.proj_name = utils.camel2snake(self.__class__.__name__, '-')
        self.embed_type = embed

        if model_name is None:
            self.model = CNNAE(loss, 3, 64, sample_size, 14, [64, 128, 256], residual="add", down_up_sample=2)
        elif model_name == "stable-diffusion":
            self.model = SDAutoEncoder(sample_size)
        elif model_name == "ms-vgg":
            self.model = MultiStageVGGAE()
        else:
            self.model = AutoEncoder(model_name, loss)

        if model_name == "ms-vgg":
            self.gaussian_ot = Sequential(*[
                GaussianOT(ae.latent_size, ae, embed, reset_target_features=False)
                for ae in self.model if isinstance(ae, AEVGG)
            ])
        else:
            self.gaussian_ot = Sequential(
                GaussianOT(self.model.latent_size, self.model, embed, reset_target_features=False)
            )

        # these transforms are actually nn.Module.
        # we remove them from the network because they mess with DDP
        # (DDP dislikes models with unused parameters/modules)
        if hasattr(self.model, 'preprocess'):
            self.preprocess = self.model.preprocess
            del self.model.preprocess
        else:
            self.preprocess = T.Compose([T.Resize((sample_size,) * 2), T.ToTensor()])
        if hasattr(self.model, 'normalize'):
            self.normalize = self.model.normalize
            del self.model.normalize
        else:
            self.normalize = T.Normalize((0.5,), (0.5,))
        if hasattr(self.model, 'denormalize'):
            self.denormalize = self.model.denormalize
            del self.model.denormalize
        else:
            self.denormalize = torch.nn.Sequential(T.Normalize((0.,), (1/0.5,)), T.Normalize((-0.5,), (1.,)))

    # def transport_img(self, img: Union[torch.Tensor, Image, str]):

    def prepare_data(
            self,
            dataset_name: str,
            degradation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            data_dir: Optional[str] = None,
            image_key: Optional[str] = None,  # relevant for hugging face dataset
            batch_size: Optional[int] = 100,  # relevant for hugging face dataset
            normalize: bool = True
    ):
        """
        Preprocess the given dataset and store the result in cache.
        It should be performed on all machines.

        :param dataset_name: torchvision dataset name (e.g. "ImageNet") or
        path to dataset in hugging-face datasets' hub (e.g. "imagenet-1k")
        :param data_dir: local cache directory of the dataset
        :param degradation: the transforms to apply as degradation (tv transforms are recommended).
        :param image_key: The filed name where the images are to be found in the dataset
        (i.e. "imagenet-1k" --> `image_key`="image", "cifar10" --> `image_key`="img")
        :param batch_size: the batch size per-process (per-gpu)
        :param normalize: whether to normalize the data
        """
        def preprocess(examples, *, preprocess_func, degrade_func, normalize_func, key=None):
            if key is None:
                img = preprocess_func(examples.convert("RGB"))
                res = {"pixel_values": normalize_func(img)}
                if degrade_func is not None:
                    res["degraded"] = normalize_func(degrade_func(img))
                return res
            res = {"pixel_values": [normalize_func(preprocess_func(img.convert("RGB"))) for img in examples[key]]}
            if degrade_func is not None:
                res["degraded"] = [normalize_func(degrade_func(preprocess_func(img.convert("RGB")))) for img in examples[key]]
            return res

        preprocess = partial(
            preprocess,
            preprocess_func=self.preprocess,
            degrade_func=degradation,
            normalize_func=self.normalize if normalize else torch.nn.Identity()
        )

        if dataset_name in torchvision.datasets.__all__:
            dataset = self._tv_dataset(dataset_name, data_dir, download=True, transforms=preprocess)
            return dataset, self._tv_collate_fn
        else:
            dataset = load_dataset(dataset_name, cache_dir=data_dir)
            preprocess = partial(preprocess, key=self._get_image_key(dataset["train"], image_key))
            dataset = dataset.map(preprocess, batch_size=batch_size, writer_batch_size=batch_size)
            dataset.set_format("torch")
            return dataset, None

    @torch.inference_mode()
    def restore(
            self,
            degradation: Callable[[torch.Tensor], torch.Tensor],
            mmse_pretrained: Optional[str] = None,
            dataset_name: str = "imagenet-1k",
            data_dir: Optional[str] = None,
            image_key: str = "image",
            batch_size: int = 100,
            num_workers: int = 10,
            n_steps: int = 1,
            n_plot_samples: int = 4,
            tex: bool = False,
            tags: Tuple[str, ...] = tuple()
    ):
        """
        Perform the unpaired, blind restoration task given by the `degradation` operator
        using latent gaussian optimal transport.

        :param degradation: the transforms to apply as degradation (tv transforms are recommended).
        :param mmse_pretrained: path to a pretrained MMSE model.
         If left unfilled, will transport the degraded images (blind setting).
        :param dataset_name: high-resolution image dataset from hugging-face datasets' hub or local path directory
        :param image_key: The field (column) where PIL images are to be found in the dataset
        (i.e. "imagenet-1k" --> `image_key`="image", "cifar10" --> `image_key`="img")
        Default behaviour is to search among {"image", "images", "img"}
        :param data_dir: local cache directory for the dataset
        :param batch_size: the batch size per-process (per-gpu)
        :param n_steps: the number of transport step
        :param n_plot_samples: the number of image per category to plot
        :param tex: if ``True`` will save the result with .pgf extension for LaTex integration
        """
        out_dir = f'logs/new-restore-{dataset_name}-{self.embed_type}'
        if len(tags): out_dir += '-' + '-'.join(tags)

        ddp = Accelerator()
        if ddp.is_local_main_process:
            if not os.path.exists(out_dir): os.mkdir(out_dir)

        cache = f'{out_dir}/.cached_resutls.pt'
        if os.path.exists(cache):
            ddp.print(f'loading cached results from {cache}')
            results = torch.load(cache)
        else:
            metrics = MetricCollection({
                "PSNR": PeakSignalNoiseRatio(),
                "FID": FrechetInceptionDistance(preprocess=self.denormalize)
            })
            mmse = MMSE.from_pretrained(mmse_pretrained) if mmse_pretrained else torch.nn.Identity()

            dataset, collate_fn = self.prepare_data(dataset_name, degradation, data_dir, image_key, batch_size)
            dl_kwargs = dict(pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)

            degraded_results, clean_results = transport(
                ddp, torch.nn.Identity(), self.gaussian_ot, dataset, metrics, batch_size, n_plot_samples,
                desc="degraded", pg_star=1, denormalize_func=self.denormalize, **dl_kwargs
            )

            for ot in self.gaussian_ot: ot.reset()

            results = [(r'$\mathbf{x}$', clean_results), (r'$\mathbf{y}$', degraded_results)]
            degraded_transported, _ = transport(
                ddp, torch.nn.Identity(), self.gaussian_ot, dataset, metrics, batch_size, n_plot_samples,
                desc="y transported", pg_star=0., denormalize_func=self.denormalize, **dl_kwargs
            )
            results.append((r'$\mathbf{\hat{y}}_{0}$', degraded_transported))

            for ot in self.gaussian_ot: ot.reset()

            for step in range(n_steps+1):
                pg_star = step / n_steps
                transported = transport(
                    ddp, mmse, self.gaussian_ot, dataset, metrics, batch_size, n_plot_samples,
                    desc=f"mmse pg_star={pg_star}", pg_star=pg_star, denormalize_func=self.denormalize, **dl_kwargs
                )[0]
                desc = rf'$\mathbf{{\hat{{x}}}}_{{{pg_star}}}$' if pg_star < 1 else r'$\mathbf{x}^*$'
                results.append((desc, transported))

            ddp.print(f'saving results in {cache}')
            ddp.save(results, cache)

        if ddp.is_local_main_process:
            self._plot_results(out_dir, results, tex)
            self._plot_results(out_dir, results, not tex)  # TODO: remove this call

    def mmse(
            self,
            degradation: Callable[[torch.Tensor], torch.Tensor],
            dataset_name: str = "imagenet-1k",
            data_dir: Optional[str] = None,
            image_key: str = "image",
            attn_max_resolution: int = 16,
            capacity: int = 48,
            epoch: int = 3,
            lr: float = 2e-4,
            batch_size: int = 16,
            num_workers: int = 10,
            logger: Optional[Literal["wandb", "tensorboard"]] = "tensorboard",
            ckpt: Optional[Union[Literal['best', 'last'], str]] = 'best',
            val_interval: Union[int, float] = 1.,
            debug: bool = False,
            tags: Tuple[str, ...] = tuple()
    ):
        """
        Train a Minimal Mean Squared Error estimator (MMSE) on the given degradations

        :param degradation: the transforms to apply as degradation (tv transforms are recommended).
        :param dataset_name: torchvision dataset name or path to dataset in hugging-face datasets' hub
        :param data_dir: local cache directory of the dataset
        :param image_key: The filed name where the images are to be found in the dataset
        (i.e. "imagenet-1k" --> `image_key`="image", "cifar10" --> `image_key`="img")
        :param attn_max_resolution: the resolutions on which to apply attention
        :param capacity: number of channels after the first convolution
        :param epoch: number of training epochs
        :param lr: training learning-rate
        :param batch_size: training batch-size
        :param num_workers: num dataloader workers
        :param logger: optional logger to follow training
        :param ckpt: path to an existing checkpoint (can also be 'best' or 'last')
        :param val_interval: frequency of validation over training epochs (can also be fraction)
        :param debug: whether to run the training in debug mode (only 1 training and validation batches)
        """
        metrics = MetricCollection({
            "PSNR": PeakSignalNoiseRatio(),
            "FID": FrechetInceptionDistance(preprocess=self.denormalize)
        }).clone(prefix='mmse/val/')

        dataset, collate_fn = self.prepare_data(dataset_name, degradation, data_dir, image_key, batch_size)
        dataload_kwargs = dict(pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)

        n_blocks = len(bin(self.model.sample_size))-len(bin(self.model.sample_size).rstrip('0'))
        latent_resolution = self.model.sample_size // 2 ** n_blocks
        n_attn_blocks = int(attn_max_resolution // latent_resolution)
        base_cap = math.log2(capacity)
        channels = np.logspace(base_cap, base_cap + n_blocks, n_blocks, False, 2, int).clip(max=2048)
        model = MMSE(
            down_block_types=("DownBlock2D",) * (n_blocks-n_attn_blocks) + ("AttnDownBlock2D",) * n_attn_blocks,
            up_block_types=("AttnUpBlock2D",) * n_attn_blocks + ("UpBlock2D",) * (n_blocks-n_attn_blocks),
            norm_num_groups=capacity//2,
            block_out_channels=channels,  # noqa
        )

        train_model(
            model=model, dataset=dataset, metric=metrics, monitor='mmse/val/PSNR', mode='max',
            early_stopping_patience=3, loggers=[logger], project_name=self.proj_name,
            options=['new', 'mmse', dataset_name, *tags], max_epoch=epoch, lr=lr, batch_size=batch_size,
            val_every=val_interval, debug=debug, train_ckpt_path=ckpt, denormalize_func=self.denormalize,
            **dataload_kwargs
        )

    def autoencoder(
            self,
            dataset_name: str = "imagenet-1k",
            data_dir: Optional[str] = None,
            image_key: str = "image",
            epoch: int = 3,
            lr: float = 2e-4,
            batch_size: int = 16,
            num_workers: int = 10,
            logger: Optional[Literal["wandb", "tensorboard"]] = "tensorboard",
            ckpt: Optional[Union[Literal['best', 'last'], str]] = 'best',
            val_interval: Union[int, float] = 1.,
            debug: bool = False,
    ):
        """
        Train theo Auto-Encoder (AE) on the given dataset

        :param dataset_name: torchvision dataset name or path to dataset in hugging-face datasets' hub
        :param data_dir: local cache directory of the dataset
        :param image_key: The filed name where the images are to be found in the dataset
        (i.e. "imagenet-1k" --> `image_key`="image", "cifar10" --> `image_key`="img")
        :param epoch: number of training epochs
        :param lr: training learning-rate
        :param batch_size: training batch-size
        :param num_workers: num dataloader workers
        :param logger: optional logger to follow training
        :param ckpt: path to an existing checkpoint (can also be 'best' or 'last')
        :param val_interval: frequency of validation over training epochs (can also be fraction)
        :param debug: whether to run the training in debug mode (only 1 training and validation batches)
        """
        ae_class = self.model.__class__.__name__
        n_params = sum(map(torch.numel, self.model.parameters()))
        print(f"training {ae_class}: {utils.human_format(n_params)} params")
        metrics = MetricCollection({
            "PSNR": PeakSignalNoiseRatio(),
            "FID": FrechetInceptionDistance(preprocess=self.denormalize)
        }).clone(prefix='autoencoder/val/')

        dataset, collate_fn = self.prepare_data(dataset_name, None, data_dir, image_key, batch_size)
        dataload_kwargs = dict(pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)

        train_model(
            self.model, dataset=dataset, metric=metrics, monitor='autoencoder/val/PSNR', mode='max',
            early_stopping_patience=3, loggers=[logger], project_name=self.proj_name,
            options=['ae', dataset_name, self.model_name], max_epoch=epoch, lr=lr, batch_size=batch_size,
            val_every=val_interval, debug=debug, train_ckpt_path=ckpt, denormalize_func=self.denormalize,
            train_params=self.model.decoder.parameters(), **dataload_kwargs
        )

    @staticmethod
    def _tv_collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = {k: torch.stack([dic[k] for dic in images]) for k in images[0]}  # list of dict to dict of list
        return images

    @staticmethod
    def _tv_dataset(dataset_name: str, root: Optional[str], download: bool, transforms: Callable):
        dataset_cls = getattr(torchvision.datasets, dataset_name)
        signature = inspect.signature(dataset_cls.__init__)
        args = signature.parameters.keys()

        # Some datasets have arg `split` and others `train`
        # e.g. ImageNet(.., split='train') and MNIST(.., train=True)
        if 'split' in args:
            _train_kwarg, _val_kwarg = dict(split='train'), dict(split='val')
        elif 'train' in args:
            _train_kwarg, _val_kwarg = dict(train=True), dict(train=False)
        else:
            _train_kwarg, _val_kwarg = {}, {}

        if 'download' in args:
            _train_kwarg['download'] = download
            _val_kwarg['download'] = download

        return {
            "train": dataset_cls(root, transform=transforms, **_train_kwarg),
            "validation": dataset_cls(root, transform=transforms, **_val_kwarg)
        }

    @staticmethod
    def _get_image_key(dataset, image_key: Optional[str] = None) -> str:
        if image_key is not None: return image_key
        for key in ["img", "image", "images"]:
            if key in dataset.column_names:
                image_key = key
                return image_key
        if image_key is None:
            raise ValueError(f"image key not found for dataset {dataset}. Please specify --image_key")

    @staticmethod
    def _plot_results(save_dir: str, results: Dict, tex: bool = False):
        def to_pil(img: torch.Tensor) -> Image:
            return fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())

        if tex:
            matplotlib.use("pgf")
            matplotlib.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'font.size': 10,
                'text.usetex': True,
                'pgf.rcfonts': True
            })

        shape = results[0][1].samples.shape
        nrows, ncols, (h, w) = shape[0], len(results), shape[-2:]
        plt.figure(figsize=(max(ncols * w / 100, 5), max(nrows * h / 100, 5) + 0.4))

        mat = gridspec.GridSpec(1, ncols, hspace=0., wspace=0.)
        for res, col in zip(results, mat):
            name, res = res
            rows = gridspec.GridSpecFromSubplotSpec(nrows, 1, subplot_spec=col, hspace=0., wspace=0.)

            for i, cell in enumerate(rows):
                ax = plt.subplot(cell)
                if i == 0:
                    ax.text(0.5, 1.17, name, size=9, ha="center", transform=ax.transAxes)
                    ax.text(0.5, 1.10, rf'$PSNR: {res.performance["PSNR"]:.2f}$', size=9, ha="center", transform=ax.transAxes)
                    ax.text(0.5, 1.03, rf'$FID: {res.performance["FID"]:.2f}$', size=9, ha="center", transform=ax.transAxes)

                ax.imshow(to_pil(res.samples[i]))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        if tex:
            plt.savefig(f'{save_dir}/transport_collage.pgf')
        else:
            plt.savefig(f'{save_dir}/transport_collage.png')
            plt.show()


if __name__ == '__main__':
    CLI(LatentDmax)
