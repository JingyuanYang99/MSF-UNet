# import imageio.v2 as imageio
# import glob
# height = '925'
# images = []
# for filename in sorted(glob.glob("/home/ub/yjy/3D wind field/fusion_3D_wind/fig/all_data_images_uvw/w/"+height+"hPa/*.png")):
#     images.append(imageio.imread(filename))
#
# imageio.mimsave("/home/ub/yjy/3D wind field/fusion_3D_wind/fig/gif/"+height+"hPa_w.gif", images, fps=8)

import imageio.v2 as imageio
import glob
height = '925'
images = []
for filename in sorted(glob.glob("/home/ub/yjy/3D wind field/fusion_3D_wind_uv/15min_wind/fig/all_data_images_uv/all_15min_frames_fixed_style/02_wind_frames/*.png")):
    images.append(imageio.imread(filename))

imageio.mimsave("/home/ub/yjy/3D wind field/fusion_3D_wind_uv/15min_wind/fig/all_data_images_uv/all_15min_frames_fixed_style/BT_wind.gif", images, fps=4)