import bpy
import os
from mathutils import Vector
import math
import sys


def load_3Dmodel(file_path):
    """
    加载3D模型文件
    
    参数:
        file_path: 模型文件的完整路径
    """
    # 确保文件存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 清空当前Blender场景
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 根据文件扩展名选择适当的导入方法
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.blend':
        # 导入.blend文件
        with bpy.data.libraries.load(file_path, link=False) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects]
            
        # 添加导入的对象到场景
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
                
    elif file_ext == '.obj':
        # 导入OBJ文件
        bpy.ops.import_scene.obj(filepath=file_path)
        
    elif file_ext == '.fbx':
        # 导入FBX文件
        bpy.ops.import_scene.fbx(filepath=file_path)
        
    elif file_ext == '.stl':
        # 导入STL文件
        bpy.ops.import_mesh.stl(filepath=file_path)
        
    elif file_ext == '.glb' or file_ext == '.gltf':
        # 导入glTF文件
        bpy.ops.import_scene.gltf(filepath=file_path)
        
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    print(f"成功导入模型: {file_path}")


def combine_objects():
    """
    尝试合并所有网格对象，或返回主要对象
    """
    # 获取所有网格对象
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not mesh_objects:
        print("未找到网格对象。请确保模型正确导入！")
        return None
    
    print(f"找到 {len(mesh_objects)} 个网格对象")
    
    # 如果只有一个网格对象(不包括我们创建的平面)
    if len(mesh_objects) == 1 or (len(mesh_objects) == 2 and "Exhibition_Floor" in [obj.name for obj in mesh_objects]):
        # 找到非地板的网格对象
        for obj in mesh_objects:
            if obj.name != "Exhibition_Floor":
                main_object = obj
                print(f"只有一个主网格对象: {main_object.name}，跳过合并步骤")
                # 应用变换
                bpy.context.view_layer.objects.active = main_object
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                return main_object
    
    # 以下是有多个网格时的合并逻辑
    # 首先应用所有对象的变换
    for obj in mesh_objects:
        if obj.name != "Exhibition_Floor":  # 不处理地板
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # 选择所有非地板网格对象用于合并
    bpy.ops.object.select_all(action='DESELECT')
    main_object = None
    
    for obj in mesh_objects:
        if obj.name != "Exhibition_Floor":
            obj.select_set(True)
            if main_object is None:
                main_object = obj
                bpy.context.view_layer.objects.active = main_object
    
    # 如果有多个对象可以合并
    if main_object and sum(1 for obj in mesh_objects if obj.name != "Exhibition_Floor") > 1:
        # 合并选中的对象
        try:
            bpy.ops.object.join()
            print("成功合并多个网格对象")
        except Exception as e:
            print(f"合并对象时出错: {e}")
    
    # 重命名合并后的对象
    if main_object:
        main_object.name = "Combined_Object"
        print(f"主要对象命名为: {main_object.name}")
    
    return main_object

def move_bbox_to_origin(obj):

    # Get the bounding box vertices of the object
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    # Calculate the center of the bounding box
    bbox_center = sum(bbox, Vector()) / 8

    # Move the object
    obj.location -= bbox_center

def setup_lights():
    """
    设置简化的灯光效果
    创建顶部主光源、环绕填充光和背景环境光，但不创建展台
    """
    # 清除现有灯光
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj)
    
    # 1. 设置顶部主光源 (区域灯)
    top_light = bpy.data.lights.new(name="Top_Area_Light", type='AREA')
    top_light_obj = bpy.data.objects.new(name="Top_Area_Light", object_data=top_light)
    bpy.context.collection.objects.link(top_light_obj)
    top_light_obj.location = (0, 0, 4)
    top_light_obj.rotation_euler = (0, 0, 0)  # 直射下方
    top_light.energy = 500
    top_light.size = 5  # 大面积柔光
    top_light.color = (1.0, 0.98, 0.95)  # 轻微暖白色
    
    # 2. 设置四周环绕的填充光 (点光源)
    fill_positions = [
        (3, 3, 2),    # 右前
        (-3, 3, 2),   # 左前
        (3, -3, 2),   # 右后
        (-3, -3, 2),  # 左后
    ]
    
    for i, pos in enumerate(fill_positions):
        fill_light = bpy.data.lights.new(name=f"Fill_Light_{i+1}", type='POINT')
        fill_light_obj = bpy.data.objects.new(name=f"Fill_Light_{i+1}", object_data=fill_light)
        bpy.context.collection.objects.link(fill_light_obj)
        fill_light_obj.location = pos
        fill_light.energy = 100
        fill_light.color = (0.9, 0.9, 1.0)  # 轻微冷色调
    
    # 3. 设置边缘光/轮廓光 (聚光灯)
    rim_positions = [
        (0, 5, 1.5, 0, -1.5, 0),   # 前方
        (0, -5, 1.5, 0, 1.5, 0),   # 后方
        (5, 0, 1.5, -1.5, 0, 0),   # 右侧
        (-5, 0, 1.5, 1.5, 0, 0),   # 左侧
    ]
    
    for i, (x, y, z, rx, ry, rz) in enumerate(rim_positions):
        rim_light = bpy.data.lights.new(name=f"Rim_Light_{i+1}", type='SPOT')
        rim_light_obj = bpy.data.objects.new(name=f"Rim_Light_{i+1}", object_data=rim_light)
        bpy.context.collection.objects.link(rim_light_obj)
        rim_light_obj.location = (x, y, z)
        rim_light_obj.rotation_euler = (rx, ry, rz)
        rim_light.energy = 150
        rim_light.spot_size = 1.2  # 聚光角度
        rim_light.spot_blend = 0.3  # 边缘过渡
        rim_light.color = (1, 1, 1)
    
    # 4. 添加环境光 (太阳灯)
    sun_light = bpy.data.lights.new(name="Sun_Light", type='SUN')
    sun_light_obj = bpy.data.objects.new(name="Sun_Light", object_data=sun_light)
    bpy.context.collection.objects.link(sun_light_obj)
    sun_light_obj.rotation_euler = (0.5, 0.5, 0.5)  # 从角落照射
    sun_light.energy = 1.0  # 低强度
    sun_light.color = (1, 1, 1)
    
    # 5. 设置渲染属性以改善质量
    bpy.context.scene.render.engine = 'CYCLES'  # 使用Cycles渲染器获得更好的光影效果
    bpy.context.scene.cycles.samples = 128  # 采样数，提高可以获得更好质量但渲染更慢
    bpy.context.scene.cycles.device = 'GPU'  # 使用GPU加速渲染
    
    # 确保存在世界对象并设置环境光遮蔽(AO)
    if bpy.context.scene.world is None:
        # 创建新的世界对象
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        print("创建了新的世界对象")
    
    # 在Blender 4.3+中，AO设置可能在不同位置
    try:
        # 尝试使用传统路径
        bpy.context.scene.world.light_settings.use_ambient_occlusion = True
        bpy.context.scene.world.light_settings.ao_factor = 0.5
    except AttributeError:
        try:
            # 在Blender 4.3+中可能使用世界节点
            bpy.context.scene.world.use_nodes = True
            # 这部分取决于Blender版本，可能需要创建适当的节点
            print("注意: 此Blender版本不支持直接设置环境光遮蔽，跳过该设置")
        except Exception as e:
            print(f"设置环境光遮蔽时出错: {e}")
    
    print("已设置灯光")

def render_views(output_dir, angle_step=30, distance=4.0, obj=None):
    """
    从360度环绕视角渲染对象，优化相机距离使模型尽可能大且完整
    
    参数:
        output_dir: 输出目录路径
        angle_step: 角度间隔，默认30度
        distance: 相机与模型的基础距离
        obj: 渲染的目标对象，用于计算最佳距离
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置相机
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object
    
    # 设置相机视场角，使用较大的视场角确保捕捉完整模型
    camera_data.lens_unit = 'FOV'
    camera_data.angle = math.radians(55)  # 稍微减小视场角，让模型更大
    
    # 计算最佳固定距离
    fixed_distance = distance
    if obj:
        # 获取对象尺寸
        dimensions = obj.dimensions
        # 计算对角线长度作为参考
        diagonal = math.sqrt(dimensions.x**2 + dimensions.y**2 + dimensions.z**2)
        
        # 使用更小的安全系数，让模型在视野中更大
        # 从2.5减小到1.8，但仍保持安全边界
        fixed_distance = max(distance, diagonal * 1.8)
        print(f"模型对角线尺寸: {diagonal:.4f}, 优化相机距离: {fixed_distance:.4f}")
    
    # 渲染360度视图，每隔angle_step度一张
    for angle_deg in range(0, 360, angle_step):
        # 将角度转换为弧度
        angle_rad = math.radians(angle_deg)
        
        # 计算相机位置（水平圆形轨道，稍微抬高视角）
        x = fixed_distance * math.cos(angle_rad)
        y = fixed_distance * math.sin(angle_rad)
        z = fixed_distance * 0.28  # 略微降低高度，使模型更居中
        
        # 设置相机位置
        camera_object.location = (x, y, z)
        
        # 使相机朝向原点（模型中心）
        direction = Vector((0, 0, 0)) - Vector(camera_object.location)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera_object.rotation_euler = rot_quat.to_euler()
        
        # 设置渲染输出文件
        bpy.context.scene.render.filepath = os.path.join(output_dir, f"angle_{angle_deg:03d}.png")
        
        # 进行渲染
        bpy.ops.render.render(write_still=True)
        print(f"角度 {angle_deg}° 渲染完成，保存至 {bpy.context.scene.render.filepath}")


def save_combined_object(output_dir, model_name):
    """
    保存合并后的对象到blend文件
    
    参数:
        output_dir: 输出目录
        model_name: 模型名称
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 构建输出文件路径
    output_file = os.path.join(output_dir, f"{model_name}_final.blend")

    # 保存当前场景为.blend文件
    bpy.ops.wm.save_as_mainfile(filepath=output_file)
    print(f"模型已保存至: {output_file}")


def normalize_model_size(obj, target_size=2.0):
    """
    规范化模型大小，确保其最大尺寸为目标大小
    
    参数:
        obj: 要规范化的对象
        target_size: 目标最大尺寸，默认为2.0单位
    """
    # 获取对象的当前尺寸
    dimensions = obj.dimensions
    max_dimension = max(dimensions.x, dimensions.y, dimensions.z)
    
    print(f"模型原始尺寸: X={dimensions.x:.4f}, Y={dimensions.y:.4f}, Z={dimensions.z:.4f}")
    
    if max_dimension == 0:
        print("警告: 对象尺寸为零，无法规范化大小")
        return
    
    # 计算缩放比例
    scale_factor = target_size / max_dimension
    
    # 应用统一缩放
    obj.scale = obj.scale * scale_factor
    
    # 应用变换使缩放生效
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # 验证新尺寸
    new_dimensions = obj.dimensions
    print(f"模型规范化后尺寸: X={new_dimensions.x:.4f}, Y={new_dimensions.y:.4f}, Z={new_dimensions.z:.4f}")
    print(f"缩放比例: {scale_factor:.4f}")


def get_model_path_from_args():
    """
    从命令行参数获取模型文件路径
    格式: blender --background --python render.py -- model_filename
    
    返回值:
        (model_path, model_name): 模型文件完整路径和不带扩展名的文件名
    """
    # 获取TRELLIS目录（脚本现在直接位于TRELLIS目录）
    trellis_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模型现在应该在TRELLIS/OUTPUTS目录中查找
    model_dir = os.path.join(trellis_dir, "OUTPUTS")
    
    # 查找"--"之后的参数
    try:
        idx = sys.argv.index("--")
        if idx < len(sys.argv) - 1:
            # 获取模型文件名
            model_filename = sys.argv[idx + 1]
            
            # 处理仅提供文件名而没有扩展名的情况
            if not os.path.splitext(model_filename)[1]:
                # 寻找匹配的文件
                potential_files = []
                for ext in ['.blend', '.obj', '.fbx', '.stl', '.glb', '.gltf']:
                    test_path = os.path.join(model_dir, f"{model_filename}{ext}")
                    if os.path.exists(test_path):
                        potential_files.append(test_path)
                
                if len(potential_files) == 1:
                    model_path = potential_files[0]
                elif len(potential_files) > 1:
                    print(f"发现多个可能的文件: {potential_files}")
                    # 优先使用GLB格式
                    glb_files = [f for f in potential_files if f.endswith('.glb')]
                    if glb_files:
                        model_path = glb_files[0]
                    else:
                        model_path = potential_files[0]  # 使用第一个找到的文件
                    print(f"使用: {model_path}")
                else:
                    raise FileNotFoundError(f"找不到匹配的文件: {model_filename}")
            else:
                # 完整文件名
                model_path = os.path.join(model_dir, model_filename)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"找不到文件: {model_path}")
            
            # 获取不带扩展名的文件名
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            return model_path, model_name
        else:
            # 没有提供模型名称参数
            raise ValueError("未提供模型名称")
    except (ValueError, IndexError, FileNotFoundError) as e:
        print(f"错误: {e}")
        print("请正确指定模型名称，例如: blender --background --python render.py -- model_name")
        print(f"模型应位于: {model_dir}")
        
        # 尝试列出OUTPUTS目录中的可用模型
        if os.path.exists(model_dir):
            models = []
            for ext in ['.blend', '.obj', '.fbx', '.stl', '.glb', '.gltf']:
                models.extend([f for f in os.listdir(model_dir) if f.endswith(ext)])
            
            if models:
                print("\n可用的模型文件:")
                for model in sorted(models):
                    print(f"  - {model}")
        
        sys.exit(1)  # 退出程序


if __name__ == "__main__":
    # 获取模型路径和名称
    model_path, model_name = get_model_path_from_args()
    print(f"渲染模型: {model_path}")
    
    # 加载3D模型
    load_3Dmodel(model_path)
    
    # 合并所有网格对象
    combined_object = combine_objects()

    # 处理模型
    if combined_object:
        # 先规范化大小
        normalize_model_size(combined_object, target_size=2.0)
        
        # 然后移动到原点
        move_bbox_to_origin(combined_object)
        
        # 恢复使用原始灯光设置
        setup_lights()

        # 设置TRELLIS目录作为输出基础目录（脚本直接位于TRELLIS目录）
        trellis_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 创建RENDERED_VIEWS目录
        rendered_views_dir = os.path.join(trellis_dir, "RENDERED_VIEWS")
        model_output_dir = os.path.join(rendered_views_dir, model_name)
        
        print(f"渲染输出将保存至: {model_output_dir}")
        
        # 进行360度渲染，传入模型对象用于计算适应性距离
        camera_distance = 4.0
        render_views(model_output_dir, angle_step=30, distance=camera_distance, obj=combined_object)

        # 保存最终模型到RENDERED_VIEWS目录
        save_combined_object(rendered_views_dir, model_name)
        
        print(f"模型 {model_name} 渲染完成")
    else:
        print("错误: 未能创建合并对象")