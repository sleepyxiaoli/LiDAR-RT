import torch
import numpy as np
import math
from typing import Optional, Dict, List, Tuple, Union

class CameraSensor:
    def __init__(self, sensor2ego: Union[np.ndarray, torch.Tensor], 
                 name: str, 
                 image_size: Tuple[int, int], 
                 intrinsic: Union[np.ndarray, torch.Tensor],
                 distortion: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 data_type: str = 'Waymo'):
        """
        Initialize a CameraSensor object.
        
        Args:
            sensor2ego: Transformation matrix from sensor to ego frame (4x4)
            name: Name of the camera sensor
            image_size: (height, width) of the camera images
            intrinsic: Camera intrinsic matrix (3x3)
            distortion: Camera distortion parameters (optional)
            data_type: Type of dataset ('Waymo' or 'KITTI')
        """
        if isinstance(sensor2ego, np.ndarray):
            sensor2ego = torch.from_numpy(sensor2ego)
        if isinstance(intrinsic, np.ndarray):
            intrinsic = torch.from_numpy(intrinsic)
        if isinstance(distortion, np.ndarray):
            distortion = torch.from_numpy(distortion)
            
        self.sensor2ego = sensor2ego.float().cpu()  # Sensor to ego transformation
        self.intrinsic = intrinsic.float().cpu()    # Camera intrinsic matrix
        self.distortion = distortion.float().cpu() if distortion is not None else None
        self.name = name
        self.data_type = data_type
        self.W, self.H = image_size
        
        # Storage for frame data
        self.sensor2world = {}      # key: frame, value: tensor(4, 4)
        self.ego2world = {}         # key: frame, value: tensor(4, 4)
        self.sensor_center = {}     # key: frame, value: tensor(3)
        self.images = {}            # key: frame, value: tensor(H, W, 3)
        self.depth_maps = {}        # key: frame, value: tensor(H, W)
        self.masks = {}            # key: frame, value: tensor(H, W) bool
        self.num_frames = 0
        
        self.train_frames = []
        self.eval_frames = []

    def add_frame(self, frame: int, 
                 ego2world: Union[np.ndarray, torch.Tensor], 
                 image: Union[np.ndarray, torch.Tensor],
                 depth_map: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 mask: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """
        Add a frame of camera data.
        
        Args:
            frame: Frame number/timestamp
            ego2world: Transformation from ego to world coordinates (4x4)
            image: Camera image (H, W, 3)
            depth_map: Optional depth map (H, W)
            mask: Optional validity mask (H, W)
        """
        if isinstance(ego2world, np.ndarray):
            ego2world = torch.from_numpy(ego2world)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
            
        # Convert to appropriate types and devices
        ego2world = ego2world.float().cpu()
        image = image.float().cpu()
        depth_map = depth_map.float().cpu() if depth_map is not None else None
        mask = mask.bool().cpu() if mask is not None else None
        
        # Compute derived quantities
        sensor2world = ego2world @ self.sensor2ego
        sensor_center = sensor2world[:3, 3]
        
        # Store data
        self.sensor2world[frame] = sensor2world
        self.ego2world[frame] = ego2world
        self.sensor_center[frame] = sensor_center
        self.images[frame] = image
        self.depth_maps[frame] = depth_map
        self.masks[frame] = mask
        self.num_frames += 1

    def set_frames(self, train_frames: List[int], eval_frames: List[int]):
        """
        Set which frames are for training and which are for evaluation.
        
        Args:
            train_frames: List of frame numbers for training
            eval_frames: List of frame numbers for evaluation
        """
        self.train_frames = train_frames
        self.eval_frames = eval_frames
        print("train:",train_frames)
        print("eval:",eval_frames)
        assert len(self.train_frames) + len(self.eval_frames) <= self.num_frames, "Illegal frame ranges!"

    def get_image(self, frame: int) -> torch.Tensor:
        """Get the image for a specific frame."""
        return self.images[frame]

    def get_depth(self, frame: int) -> Optional[torch.Tensor]:
        """Get the depth map for a specific frame."""
        return self.depth_maps.get(frame, None)

    def get_mask(self, frame: int) -> Optional[torch.Tensor]:
        """Get the validity mask for a specific frame."""
        return self.masks.get(frame, None)

    def get_intrinsics(self) -> torch.Tensor:
        """Get the camera intrinsic matrix."""
        return self.intrinsic

    def get_distortion(self) -> Optional[torch.Tensor]:
        """Get the camera distortion parameters."""
        return self.distortion

    def project_points(self, frame: int, points_3d: torch.Tensor) -> torch.Tensor:
        """
        Project 3D points in world coordinates to 2D image coordinates.
        
        Args:
            frame: Frame number
            points_3d: Tensor of 3D points in world coordinates (N, 3)
            
        Returns:
            points_2d: Tensor of 2D image coordinates (N, 2)
            depths: Tensor of depths (N,)
        """
        # Transform points to camera coordinates
        world2camera = self.sensor2world[frame].inverse()
        points_hom = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=1)
        points_camera = (world2camera @ points_hom.T).T[:, :3]
        
        # Project using intrinsics
        points_image = (self.intrinsic @ points_camera.T).T
        points_2d = points_image[:, :2] / points_image[:, 2:]
        depths = points_camera[:, 2]
        
        return points_2d, depths

    def unproject_depth(self, frame: int) -> torch.Tensor:
        """
        Unproject depth map to 3D points in world coordinates.
        
        Args:
            frame: Frame number
            
        Returns:
            points_3d: Tensor of 3D points in world coordinates (H, W, 3)
        """
        if frame not in self.depth_maps or self.depth_maps[frame] is None:
            raise ValueError(f"No depth map available for frame {frame}")
            
        depth_map = self.depth_maps[frame]
        intrinsic_inv = self.intrinsic.inverse()
        
        # Create grid of pixel coordinates
        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
        pixel_coords = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).float()
        
        # Transform to camera coordinates
        camera_coords = (intrinsic_inv @ pixel_coords.reshape(-1, 3).T).T
        camera_coords = camera_coords.reshape(self.H, self.W, 3)
        camera_coords = camera_coords * depth_map.unsqueeze(-1)
        
        # Transform to world coordinates
        camera_coords_hom = torch.cat([
            camera_coords, 
            torch.ones(self.H, self.W, 1)], dim=-1)
        world_coords = (self.sensor2world[frame] @ camera_coords_hom.reshape(-1, 4).T).T
        world_coords = world_coords[:, :3].reshape(self.H, self.W, 3)
        
        return world_coords

    def get_range_rays(self, frame: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for each pixel in the image.
        
        Args:
            frame: Frame number
            
        Returns:
            rays_o: Ray origins (H, W, 3)
            rays_d: Ray directions (H, W, 3)
        """
        K = self.intrinsic.cuda()  # [3, 3] 内参矩阵
        sensor2world = self.sensor2world[frame].cuda()  # [4, 4] 外参矩阵
        sensor_center = self.sensor_center[frame].cuda()  # [3,] 相机中心

        rays_o = sensor_center[None, None, ...].expand(self.H, self.W, 3)  # [H, W, 3]
        # print("ray_o",rays_o)
        
        u = torch.arange(self.W, device='cuda', dtype=torch.float32) + 0.5  # [W]
        v = torch.arange(self.H, device='cuda', dtype=torch.float32) + 0.5  # [H]
        # print("u",u)
        # print("v",v)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')  # [H, W]
        ones = torch.ones_like(grid_u)
        pixel_coords = torch.stack([grid_u, grid_v, ones], dim=-1)  # [H, W, 3]
        # print("pixel_coords",pixel_coords)

        rays_d_cam = pixel_coords @ torch.inverse(K).T  # [H, W, 3]

        rays_d_cam = torch.stack([
            rays_d_cam[..., 2],  # x前 = 原始z
            -rays_d_cam[..., 0], # y左 = -原始x
            -rays_d_cam[..., 1]   # z上 = -原始y
        ], dim=-1)
        
        # 转换到世界坐标系
        rays_d = rays_d_cam @ sensor2world[:3, :3].T  # 仅旋转
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # 单位化

        return rays_o, rays_d  # [H, W, 3], [H, W, 3]
    
    def undistort_image(self, frame: int) -> torch.Tensor:
        """
        Undistort the image using the camera's distortion parameters.
        
        Args:
            frame: Frame number
            
        Returns:
            undistorted_image: Undistorted image (H, W, 3)
        """
        if self.distortion is None:
            return self.images[frame]
            
        # TODO: Implement actual undistortion using OpenCV or similar
        # This is a placeholder implementation
        return self.images[frame]