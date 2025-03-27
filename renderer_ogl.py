from OpenGL import GL as gl
import util
import util_gau
import numpy as np

try:
    from OpenGL.raw.WGL.EXT.swap_control import wglSwapIntervalEXT
except:
    wglSwapIntervalEXT = None


_sort_buffer_xyz = None
_sort_buffer_gausid = None  # used to tell whether gaussian is reloaded

def _sort_gaussian_cpu(gaus, view_mat):
    xyz = np.asarray(gaus.xyz)
    view_mat = np.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    return index


def _sort_gaussian_cupy(gaus, view_mat):
    import cupy as cp
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = cp.asarray(gaus.xyz)
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = cp.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = cp.argsort(depth)
    index = index.astype(cp.int32).reshape(-1, 1)

    index = cp.asnumpy(index) # convert to numpy
    return index


def _sort_gaussian_torch(gaus, view_mat):
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = torch.tensor(gaus.xyz).cuda()
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = torch.tensor(view_mat).cuda()
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = torch.argsort(depth)
    index = index.type(torch.int32).reshape(-1, 1).cpu().numpy()
    return index


# Decide which sort to use
_sort_gaussian = None
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
    print("Detect torch cuda installed, will use torch as sorting backend")
    _sort_gaussian = _sort_gaussian_torch
except ImportError:
    try:
        import cupy as cp
        print("Detect cupy installed, will use cupy as sorting backend")
        _sort_gaussian = _sort_gaussian_cupy
    except ImportError:
        _sort_gaussian = _sort_gaussian_cpu


class GaussianRenderBase:
    def __init__(self):
        self.gaussians = None
        self._reduce_updates = True

    @property
    def reduce_updates(self):
        return self._reduce_updates

    @reduce_updates.setter
    def reduce_updates(self, val):
        self._reduce_updates = val
        self.update_vsync()

    def update_vsync(self):
        print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        raise NotImplementedError()
    
    def sort_and_update(self):
        raise NotImplementedError()

    def set_scale_modifier(self, modifier: float):
        raise NotImplementedError()
    
    def set_render_mod(self, mod: int):
        raise NotImplementedError()
    
    def update_camera_pose(self, camera: util.Camera):
        raise NotImplementedError()

    def update_camera_intrin(self, camera: util.Camera):
        raise NotImplementedError()
    
    def draw(self):
        raise NotImplementedError()
    
    def set_render_reso(self, w, h):
        raise NotImplementedError()


class OpenGLRenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        gl.glViewport(0, 0, w, h)
        self.program = util.load_shaders('shaders/gau_vert.glsl', 'shaders/gau_frag.glsl')

        # Vertex data for a quad
        self.quad_v = np.array([
            -1,  1,
            1,  1,
            1, -1,
            -1, -1
        ], dtype=np.float32).reshape(4, 2)
        self.quad_f = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32).reshape(2, 3)
        
        # load quad geometry
        vao, buffer_id = util.set_attributes(self.program, ["position"], [self.quad_v])
        util.set_faces_tovao(vao, self.quad_f)
        self.vao = vao
        self.gau_bufferid = None
        self.index_bufferid = None
        # opengl settings
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.update_vsync()
    def get_splatted_coords(self, screen_pos, camera):
        """根据屏幕坐标计算高斯splatting后的3D坐标"""
        # 将屏幕坐标转换为NDC坐标
        x_ndc = 2.0 * screen_pos[0] / camera.w - 1.0
        y_ndc = 1.0 - 2.0 * screen_pos[1] / camera.h
        
        # 获取视图和投影矩阵
        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_project_matrix()
        
        # 计算逆变换
        inv_view_proj = np.linalg.inv(proj_mat @ view_mat)
        
        # 创建齐次坐标
        near_point = np.array([x_ndc, y_ndc, -1.0, 1.0])  # 近平面
        far_point = np.array([x_ndc, y_ndc, 1.0, 1.0])     # 远平面
        
        # 转换为世界坐标
        near_world = inv_view_proj @ near_point
        near_world /= near_world[3]
        far_world = inv_view_proj @ far_point
        far_world /= far_world[3]
        
        # 计算射线方向
        ray_dir = far_world[:3] - near_world[:3]
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # 这里简化处理，实际应根据高斯分布计算加权平均坐标
        # 在实际应用中，您可能需要从着色器中获取更精确的结果
        
        # 简单返回射线上的一个点(近平面+一定距离)
        splatted_pos = near_world[:3] + ray_dir * 0.5
        
        return splatted_pos
        
    def get_splatted_world_coords(self, screen_pos, camera):
        """将屏幕坐标逆向投影到世界坐标系（考虑高斯 splatting 的影响）"""
        # 1. 转换为NDC坐标
        x_ndc = 2.0 * screen_pos[0] / camera.w - 1.0
        y_ndc = 1.0 - 2.0 * screen_pos[1] / camera.h
        
        # 2. 获取逆变换矩阵
        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_project_matrix()
        inv_view_proj = np.linalg.inv(proj_mat @ view_mat)
        
        # 3. 创建近平面和远平面的齐次坐标
        near_clip = np.array([x_ndc, y_ndc, -1.0, 1.0])  # OpenGL默认近平面z=-1
        far_clip = np.array([x_ndc, y_ndc, 1.0, 1.0])    # 远平面z=1
        
        # 4. 转换为世界坐标
        near_world = (inv_view_proj @ near_clip)[:3]  # 取xyz，忽略齐次分量
        far_world = (inv_view_proj @ far_clip)[:3]
        
        # 5. 计算射线方向（从相机指向点击位置）
        ray_origin = camera.position  # 相机在世界坐标系的位置
        ray_dir = far_world - near_world
        ray_dir /= np.linalg.norm(ray_dir)
        
        # 6. 模拟高斯 splatting 的投影深度（实际需结合着色器深度缓冲）
        # 这里简化为取射线与高斯分布的交点（假设在中间深度）
        splatted_depth = 0.5 * (near_world + far_world)
    
        return splatted_depth  # 返回世界坐标系下的坐标
    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)
        else:
            print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.gaussians = gaus
        # load gaussian geometry
        gaussian_data = gaus.flat()
        self.gau_bufferid = util.set_storage_buffer_data(self.program, "gaussian_data", gaussian_data, 
                                                         bind_idx=0,
                                                         buffer_id=self.gau_bufferid)
        util.set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

    def sort_and_update(self, camera: util.Camera):
        index = _sort_gaussian(self.gaussians, camera.get_view_matrix())
        self.index_bufferid = util.set_storage_buffer_data(self.program, "gi", index, 
                                                           bind_idx=1,
                                                           buffer_id=self.index_bufferid)
        return
   
    def set_scale_modifier(self, modifier):
        util.set_uniform_1f(self.program, modifier, "scale_modifier")

    def set_render_mod(self, mod: int):
        util.set_uniform_1int(self.program, mod, "render_mod")

    def set_render_reso(self, w, h):
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self, camera: util.Camera):
        view_mat = camera.get_view_matrix()
        util.set_uniform_mat4(self.program, view_mat, "view_matrix")
        util.set_uniform_v3(self.program, camera.position, "cam_pos")

    def update_camera_intrin(self, camera: util.Camera):
        proj_mat = camera.get_project_matrix()
        util.set_uniform_mat4(self.program, proj_mat, "projection_matrix")
        util.set_uniform_v3(self.program, camera.get_htanfovxy_focal(), "hfovxy_focal")

    def draw(self):
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        num_gau = len(self.gaussians)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(self.quad_f.reshape(-1)), gl.GL_UNSIGNED_INT, None, num_gau)
