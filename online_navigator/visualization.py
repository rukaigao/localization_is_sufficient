"""Lightweight trajectory/PF GIF rendering for online runs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_background(background_tif, max_side, out_dir: Path):
    if background_tif is None:
        return None
    path=Path(background_tif)
    if not path.exists():
        print(f"[visualization] background not found: {path}; GIF will use a blank map")
        return None
    # These are trusted local geospatial rasters, not arbitrary web uploads.
    Image.MAX_IMAGE_PIXELS=None
    im=Image.open(path).convert("RGB")
    scale=min(1.0,float(max_side)/max(im.size))
    if scale<1.0:
        im=im.resize((max(1,int(round(im.width*scale))),max(1,int(round(im.height*scale)))),Image.Resampling.LANCZOS)
    preview=out_dir/"background_preview.jpg"
    im.save(preview,quality=86,optimize=True)
    print(f"[visualization] cached background preview: {preview} ({im.width}x{im.height})")
    return im


def save_online_gif(out_path, snapshots, route_points, goal, ref_map, cfg, background_tif=None):
    if not snapshots:
        return None
    out_path=Path(out_path); out_path.parent.mkdir(parents=True,exist_ok=True)
    base=_load_background(background_tif,int(cfg.gif_max_side_px),out_path.parent)
    if base is None:
        side=min(int(cfg.gif_max_side_px),1000)
        base=Image.new("RGB",(side,side),(242,242,242))
    W,H=base.size
    lo,hi=ref_map.map_bounds(); span=np.maximum(hi-lo,1e-9)
    def xy(p):
        p=np.asarray(p,dtype=float)
        x=(p[0]-lo[0])/span[0]*(W-1)
        y=(p[1]-lo[1])/span[1]*(H-1)
        return (int(round(x)),int(round(y)))

    route=[xy(p) for p in np.asarray(route_points)]
    goal_xy=xy(goal)
    frames=[]
    stride=max(1,int(cfg.gif_frame_stride))
    draw_snaps=snapshots[::stride]
    if draw_snaps[-1] is not snapshots[-1]: draw_snaps.append(snapshots[-1])
    for snap in draw_snaps:
        im=base.copy(); d=ImageDraw.Draw(im,"RGBA")
        if len(route)>=2: d.line(route,fill=(20,120,255,190),width=max(2,W//400))
        gx,gy=goal_xy; gr=max(5,W//120)
        d.ellipse((gx-gr,gy-gr,gx+gr,gy+gr),fill=(255,215,0,220),outline=(20,20,20,255),width=2)
        # Trajectory so far
        traj=[xy(p) for p in snap["trajectory"]]
        if len(traj)>=2: d.line(traj,fill=(255,70,60,230),width=max(2,W//350))
        # PF particles, optionally thinned only for rendering.
        pos=np.asarray(snap["particles"],dtype=float)
        if len(pos)>int(cfg.gif_particle_max_draw):
            idx=np.linspace(0,len(pos)-1,int(cfg.gif_particle_max_draw),dtype=int); pos=pos[idx]
        # Render PF particles in red with a slightly larger marker so they
        # remain visible over predominantly green aerial imagery.
        r=max(2,W//500)
        for p in pos:
            x,y=xy(p); d.ellipse((x-r,y-r,x+r,y+r),fill=(255,35,35,150))
        mx,my=xy(snap["pf_mean"]); mr=max(4,W//180)
        d.ellipse((mx-mr,my-mr,mx+mr,my+mr),fill=(0,0,0,230),outline=(255,255,255,255),width=2)
        tx,ty=xy(snap["true_pos"]); tr=max(5,W//140)
        d.ellipse((tx-tr,ty-tr,tx+tr,ty+tr),fill=(255,40,40,240),outline=(255,255,255,255),width=2)
        text=(f"step {snap['step']}  {snap['state']}  risk={int(snap['risk'])}  action={snap['action']}\n"
              f"R90={snap['r90'] if snap['r90'] is not None else float('nan'):.1f}  "
              f"ESS={snap['ess_pre']:.0f}  Uobs={snap['u_obs']:.3f}  rho={snap['rho']:.3f}\n"
              f"PF robust-center error={snap['pf_mean_error']:.1f}px  goal mass={snap['goal_mass']:.3f}")
        # Small opaque text panel for readability.
        bbox=d.multiline_textbbox((8,8),text,spacing=3)
        d.rounded_rectangle((4,4,bbox[2]+12,bbox[3]+12),radius=5,fill=(255,255,255,210))
        d.multiline_text((8,8),text,fill=(0,0,0,255),spacing=3)
        frames.append(im.convert("P",palette=Image.Palette.ADAPTIVE,colors=192))
    frames[0].save(out_path,save_all=True,append_images=frames[1:],duration=int(cfg.gif_duration_ms),loop=0,optimize=False)
    print(f"[visualization] saved online PF GIF: {out_path} ({len(frames)} frames)")
    return out_path
