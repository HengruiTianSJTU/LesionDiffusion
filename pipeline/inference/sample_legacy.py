import argparse, os, sys, glob, datetime, yaml, json, pathlib as pb, re
import torch
import time
import numpy as np, nibabel as nib
from tqdm import trange
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from torch.utils.data import DataLoader
from einops import rearrange
from torchvision.utils import make_grid
from tqdm import tqdm
from functools import reduce
from collections import namedtuple
from scipy.ndimage import sobel
from inference.utils import find_vacancy, custom_to_pil, custom_to_np, logs2pil

rescale = lambda x: (x + 1.) / 2.

def combine_mask_and_im(x, overlay_coef=0.2, n_classes=10):
    # 2 h w d
    def find_mask_boundaries_3d(im, mask, color):
        boundaries = torch.zeros_like(mask)
        for i in tqdm(range(1, 12), desc="paint array"):
            m = (mask == i).numpy()
            sobel_x = sobel(m, axis=0, mode='constant')
            sobel_y = sobel(m, axis=1, mode='constant')
            sobel_z = sobel(m, axis=2, mode='constant')

            boundaries = torch.from_numpy((np.abs(sobel_x) + np.abs(sobel_y) + np.abs(sobel_z)) * i) * (boundaries == 0) + boundaries * (boundaries != 0)
        im = color[boundaries.long()] * (boundaries[..., None] > 0) + im * (boundaries[..., None] == 0)
        return im
    
    image, mask = 255 * x[0, ..., None].clamp(0, 1).repeat(1, 1, 1, 3), x[1] * 11
    mask[mask == 255] = 11
    OrganClass = namedtuple("OrganClass", ["label_name", "totalseg_id", "color"])
    abd_organ_classes = [
        OrganClass("unlabeled", 0, (0, 0, 0)),
        OrganClass("spleen", 1, (0, 80, 100)),
        OrganClass("kidney_left", 2, (119, 11, 32)),
        OrganClass("kidney_right", 3, (119, 11, 32)),
        OrganClass("liver", 5, (250, 170, 30)),
        OrganClass("stomach", 6, (220, 220, 0)),
        OrganClass("pancreas", 10, (107, 142, 35)),
        OrganClass("small_bowel", 55, (255, 0, 0)),
        OrganClass("duodenum", 56, (70, 130, 180)),
        OrganClass("colon", 57, (0, 0, 255)),
        OrganClass("urinary_bladder", 104, (0, 255, 255)),
        OrganClass("colorectal_cancer", 255, (0, 255, 0))
    ]
    colors = torch.from_numpy(np.array([a.color for a in abd_organ_classes]))
    colored_mask = (colors[mask.long()] * (mask[..., None] > 0) + image * (mask[..., None] == 0))
    colored_im = colored_mask * overlay_coef + image * (1-overlay_coef)
    colored_im = rearrange(find_mask_boundaries_3d(colored_im, mask, colors), "d ... c -> d c ...")
    return colored_im


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

@ torch.no_grad()
def sample_cond(model, instance, scale=1., n_samples=1, ddim_steps=100, ddim_eta=1, vanilla=False,
                sample_path="/mnt/workspace/dailinrui/code/multimodal/latentdiffusion/samples",
                postpone_metric_computation=False, wholemask=None):
    # metrics = compute_metrics(torch.clamp(gt*1.5, 0, 1), torch.clamp(gt*1.5, 0, 1))
    prompt = instance.get("text", None)
    sampler = DDIMSampler(model)
    shape = (1, 64, 128, 128)  # cdhw/chw, for output shape
    # with model.ema_scope():
    #     gt = model.get_input(instance, "mask")[0]
    #     # uc = model.get_learned_conditioning(n_samples * [""]) if scale != 1. else None
    #     c = model.get_learned_conditioning(n_samples * prompt)
    #     if not vanilla:
    #         input_shape = shape
    #         samples, _ = sampler.sample(S=ddim_steps,
    #                                     dims=len(shape) - 1,
    #                                     conditioning=c,
    #                                     batch_size=len(prompt) * n_samples,
    #                                     shape=input_shape,
    #                                     verbose=False,
    #                                     # unconditional_guidance_scale=scale,
    #                                     # unconditional_conditioning=uc,
    #                                     eta=ddim_eta)
    #     else:
    #         output_shape = shape
    #         samples, _ = model.p_sample_loop(c, (len(prompt) * n_samples,) + output_shape,
    #                                return_intermediates=True, verbose=True)

    #     pred = model.decode_first_stage(samples)
    #     pred = torch.clamp((pred + 1.0)/ 2.0, min=0.0, max=1.0)
            
    with model.ema_scope():
        # wholemask = model.get_input(instance, "wholemask")[0].cuda()
        gt = instance.get("wholeimage").permute(0, 4, 1, 2, 3).cuda()
        # wholemask = nib.load("/mnt/workspace/dailinrui/data/pretrained/ccdm/final_ernie_resize_to_64x128x128_w1/eval/RJ202302171638329080/pred.nii.gz").dataobj[:].transpose(2, 1, 0)
        # wholemask = torch.rot90(torch.from_numpy(zoom(wholemask, np.array((96, 512, 512)) / np.array(wholemask.shape), order=0)), dims=(1, 2), k=3)[None, None].cuda() / 255.
        wholemask = wholemask.permute(0, 4, 1, 2, 3).cuda()
        start_layer, end_layer = torch.where(wholemask.sum((0, 1, 3, 4)))[0][[0, -1]]
        shape = (4, 64, 64)
        assert wholemask.shape[0] == 1,  "batch size should be 1"
        samples = torch.zeros((n_samples,) + wholemask.shape[1:], dtype=torch.float32, device=wholemask.device)
        gen_mask = wholemask.repeat(n_samples, 1, 1, 1, 1)

        for m_ in tqdm(range(start_layer.item()-1, end_layer.item()+1), desc="slicewise image creation"):
            concat_cond = torch.cat([samples[:, :, max(0, m_ - 1)],
                                     gen_mask[:, :, m_]], axis=1)
            c = model.get_learned_conditioning(concat_cond)
            s, _ = sampler.sample(S=ddim_steps,
                                    dims=len(shape) - 1,
                                    conditioning=c,
                                    batch_size=n_samples,
                                    shape=shape,
                                    verbose=False,
                                    # unconditional_guidance_scale=scale,
                                    # unconditional_conditioning=uc,
                                    )#eta=ddim_eta)  # (num_images bs d h w)
            ds = model.decode_first_stage(s)
            samples[:, :, m_] = (ds - ds.min()) / (ds.max() - ds.min())
            Image.fromarray(make_grid(255 * torch.cat([concat_cond[:, 0:1], concat_cond[:, 1:] * 20, samples[:,:, m_]], dim=1).cpu().permute(1, 0, 2, 3), padding=5).permute(1, 2, 0).numpy().astype(np.uint8)).save(f"./samples/layers/{m_}.png")
        pred = torch.cat([samples, gen_mask], dim=1)
    
    # gt = instance["wholeimage"]
    # b, d, *shp = gt.shape
    # gt = torch.nn.functional.interpolate(gt, (512, 512), mode="bilinear")
    # samples = torch.zeros((b, d - 2, 512, 512), dtype=gt.dtype).cuda()
    # for image_mid_index in tqdm(range(d - 2), desc="slicewise autoencoder embedding"):
    #     concat_cond = gt[:, image_mid_index: image_mid_index + 3].cuda()
    #     xrec, _ = model(concat_cond)
    #     samples[:, image_mid_index - 1] = xrec[:, 1]
    # pred = torch.nn.functional.interpolate(samples, shp, mode="bilinear")[None]
    # pred = (pred - pred.min()) / (pred.max() - pred.min())
        
    if postpone_metric_computation: to_compute = []
    else: to_compute = ["lpips", "fvd"]
    metrics = compute_metrics(pred, pred, to_compute)

    if len(pred.shape) == 5: # 3d
        for ix, x_sample in enumerate(pred):
            c, *_shp = x_sample.shape
            if c == 1: 
                x_sample = 255. * x_sample.cpu().permute(1, 0, 2, 3)
                grid = make_grid(x_sample, nrow=8, padding=5).permute(1, 2, 0).numpy().astype(np.uint8)
                if x_sample.shape[0] > 100:
                    import SimpleITK as sitk
                    sitk.WriteImage(sitk.GetImageFromArray(x_sample[:, 0].numpy().astype(np.float32)),
                                    find_vacancy(os.path.join(sample_path, "sample.nii.gz")))
                else:
                    Image.fromarray(
                        grid
                    ).save(find_vacancy(os.path.join(sample_path, f"sample.png")))
                
            if c == 2: 
                x_sample = combine_mask_and_im(x_sample.cpu())
                grid = make_grid(x_sample, nrow=8, padding=5).permute(1, 2, 0).numpy().astype(np.uint8)
                Image.fromarray(
                    grid
                ).save(find_vacancy(os.path.join(sample_path, f"sample.png")))
            # x_sample = 255. * rearrange(x_sample.cpu(), 'c d h w -> (c d) 1 h w')
    if len(pred.shape) == 4: # 2d
        for x_sample in pred:
            grid = 255 * rearrange(x_sample.cpu().numpy(), 'c h w -> (c h) w 1').astype(np.uint8)
            Image.fromarray(
                grid
            ).save(find_vacancy(os.path.join(sample_path, f"sample.png")))
    metrics["pred"] = pred.cpu().numpy()
    metrics["gt"] = gt.cpu().numpy()
    metrics["cond"] = prompt
    metrics["grid"] = grid[None]
    return metrics
                

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=1,
        nplog=None, dataloader=None, postpone_metric_computation=False, stage_1_results=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    if hasattr(model, "cond_stage_model") and model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
        assert dataloader is not None
        logs = []
        nppath = os.path.join(nplog, f"all_samples.npz")
        # if not os.path.exists(nppath):
        if True:
            if stage_1_results is not None:
                stage_1_results_fs = os.listdir(stage_1_results)
                for item in stage_1_results_fs:
                    mask = nib.load(os.path.join(stage_1_results, item)).dataobj[:]
                    logs.append(sample_cond(model, batch, 
                                            n_samples=n_samples, ddim_steps=custom_steps, ddim_eta=eta,
                                            vanilla=vanilla, postpone_metric_computation=postpone_metric_computation, wholemask=mask))
            else:
                for batch in tqdm(dataloader, desc="validation loader"):
                    logs.append(sample_cond(model, batch, 
                                            n_samples=n_samples, ddim_steps=custom_steps, ddim_eta=eta,
                                            vanilla=vanilla, postpone_metric_computation=postpone_metric_computation, wholemask=batch["wholemask"]))
            npz_files = dict(pred=np.concatenate([l["pred"] for l in logs], axis=0),
                            gt=np.concatenate([l["gt"] for l in logs], axis=0),
                            grid=np.concatenate([l["grid"] for l in logs], axis=0),
                            cond=np.array(reduce(lambda x, y: x+ y, [l["cond"] for l in logs])))
            torch.cuda.empty_cache()
            np.savez(nppath, **npz_files)
        else:
            npz_files = np.load(nppath)
        
        # if postpone_metric_computation:
        #     metrics = compute_metrics(npz_files["pred"], npz_files["gt"], batch_per_segment=3)
        # with open(nppath.replace("all_samples.npz", "all_metrics.json"), "w") as f:
        #     json.dump(metrics, f, indent=4, ensure_ascii=False)
        os.makedirs(os.path.join(os.path.dirname(nppath), "pngs"), exist_ok=True)
        for ig, g in enumerate(npz_files["grid"]):
            Image.fromarray(g).save(os.path.join(os.path.dirname(nppath), "pngs", f"{ig}.png"))

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=1
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=1
    )
    parser.add_argument(
        "--inputs",
        type=str,
        help="the inputs folder from stage 1",
        default=None
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


@torch.no_grad()
def compute_metrics(pred, gt, metrics=["lpips", "fvd"], batch_per_segment=None):
    results = dict()
    if len(metrics) == 0: return results
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(pred)
    if not isinstance(gt, torch.Tensor):
        gt = torch.from_numpy(gt)
    pred, gt = pred.cuda(), gt.cuda()
        
    b, c, *shp = pred.shape
    if batch_per_segment is None: batch_per_segment = b
    assert pred.shape == gt.shape
    if c != 1:
        pred = rearrange(pred, "b c ... -> (b c) 1 ...")
        gt = rearrange(gt, "b c ... -> (b c) 1 ...")
    
    for segment in tqdm(range(0, b, batch_per_segment)):
        _pred = pred[segment: segment + batch_per_segment]
        _gt = gt[segment: segment + batch_per_segment]
        
        if "lpips" in metrics:
            from ldm.modules.losses.lpips import LPIPS
            
            # mean of 3 views
            def _compute(x, y):
                perceptual = LPIPS().to(x.device).eval()
                if len(x.shape) == 5:
                    lpips_x = perceptual(rearrange(x, "b c d h w -> (b d) c h w"),
                                        rearrange(y, "b c d h w -> (b d) c h w")).mean()
                    lpips_y = perceptual(rearrange(x, "b c d h w -> (b h) c d w"),
                                        rearrange(y, "b c d h w -> (b h) c d w")).mean()
                    lpips_z = perceptual(rearrange(x, "b c d h w -> (b w) c d h"),
                                        rearrange(y, "b c d h w -> (b w) c d h")).mean()
                    lpips_val = (lpips_x + lpips_y + lpips_z) / 3
                else:
                    lpips_val = perceptual(x, y)
                return lpips_val
            
            results["lpips"] = results.get("lpips", 0) + _compute(_pred, _gt) * batch_per_segment / b
            
        if "fvd" in metrics:
            from scripts.fvd import compute_fvd
            
            assert len(_pred.shape) == 5
            assert b > 1
            try:
                results["fvd"] = results.get("fvd", 0) + \
                    compute_fvd(rearrange(_pred, "b c d h w -> b d h w c").repeat(1, 1, 1, 1, 3),
                                rearrange(_gt, "b c d h w -> b d h w c").repeat(1, 1, 1, 1, 3)) * batch_per_segment / b
            except Exception as e:
                results["fvd"] = results.get("fvd", 0) * (b) / (b-1)
            
    return results


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-2])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "test_samples")
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(logdir, exist_ok=True)
    # os.makedirs(numpylogdir, exist_ok=True)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)
    
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    dataloader = DataLoader(data.datasets["validation"], batch_size=opt.batch_size, shuffle=False)

    run(model, imglogdir, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=logdir, dataloader=dataloader, postpone_metric_computation=True, stage_1_results=opt.inputs)

    print("done.")
