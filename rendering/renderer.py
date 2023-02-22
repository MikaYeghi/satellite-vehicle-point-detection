import torch
from torch import nn
import torchvision.transforms as tv_transf

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    DirectionalLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    BlendParams
)

import pdb

class Renderer(nn.Module):
    def __init__(self, device):
        super().__init__()
        # self.avg_pool = nn.AvgPool2d(5, stride=5)
        self.device = device
    
    def render(self, mesh, background_image, elevation, azimuth, lights_direction, distance=5.0, 
               scaling_factor=0.85, image_size=250, blur_radius=0.0, faces_per_pixel=1, intensity=0.3, ambient_color=((0.05, 0.05, 0.05),)):
        # transform = tv_transf.Resize((250, 250))
        # background_image = transform(background_image).permute(1, 2, 0)
        background_image = background_image.permute(1, 2, 0)
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        scale_xyz = scaling_factor * torch.tensor([1.0, 1.0, 1.0], device=self.device).unsqueeze(0)
        cameras = FoVOrthographicCameras(
            device=self.device, 
            R=R, 
            T=T, 
            scale_xyz=scale_xyz
        )
        
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=blur_radius, 
            faces_per_pixel=faces_per_pixel, 
        )
        
        diffuse_color = intensity * torch.tensor([1.0, 1.0, 1.0], device=self.device).unsqueeze(0)
        lights = DirectionalLights(device=self.device, direction=lights_direction, ambient_color=ambient_color, diffuse_color=diffuse_color)
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=background_image)
            )
        )

        images = renderer(mesh, lights=lights, cameras=cameras)
        images = images[0, ..., :3]
        return images
    
    def render_batch(self, meshes, background_images, elevations, azimuths, light_directions, distances,
                     scaling_factors, intensities, image_size=250, blur_radius=0.0, faces_per_pixel=1, ambient_color=((0.05, 0.05, 0.05),)):
        transform = tv_transf.Resize((250, 250))
        background_images = transform(background_images).permute(0, 2, 3, 1)
        R, T = look_at_view_transform(dist=distances, elev=elevations, azim=azimuths)
        cameras = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
            scale_xyz=scaling_factors
        )
        
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=blur_radius, 
            faces_per_pixel=faces_per_pixel, 
        )
        
        lights = DirectionalLights(device=self.device, direction=light_directions, ambient_color=ambient_color, diffuse_color=intensities)
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=background_images)
            )
        )
        
        images = renderer(meshes, lights=lights, cameras=cameras)
        images = images.permute(0, 3, 1, 2)
        images = self.avg_pool(images)
        images = images[:, :3, ...]
        
        return images