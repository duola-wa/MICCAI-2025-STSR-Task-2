import torch
import torch.nn.functional as F


def transform_points(points, transform):
    """原有的点云变换函数（保持不变）"""
    points_h = F.pad(points, (0, 1), mode='constant', value=1.0)
    transformed_points_h = torch.bmm(points_h, transform.transpose(1, 2))
    return transformed_points_h[:, :, :3]


def chamfer_distance(pred_points, gt_points):
    """
    计算Chamfer Distance
    Args:
        pred_points: [B, N, 3] 预测变换后的点云
        gt_points: [B, N, 3] 真实变换后的点云
    Returns:
        chamfer_dist: 标量，Chamfer距离
    """
    # 计算所有点对之间的距离矩阵
    pred_expanded = pred_points.unsqueeze(2)  # [B, N, 1, 3]
    gt_expanded = gt_points.unsqueeze(1)  # [B, 1, N, 3]

    dist_matrix = torch.sum((pred_expanded - gt_expanded) ** 2, dim=-1)  # [B, N, N]

    # 找到最近邻距离
    dist_pred_to_gt = torch.min(dist_matrix, dim=2)[0]  # [B, N]
    dist_gt_to_pred = torch.min(dist_matrix, dim=1)[0]  # [B, N]

    # Chamfer Distance = 双向最近邻距离的平均
    chamfer_dist = torch.mean(dist_pred_to_gt) + torch.mean(dist_gt_to_pred)

    return chamfer_dist


def decompose_transform_matrix(transform_matrix):
    """
    分解变换矩阵为旋转和平移部分
    Args:
        transform_matrix: [B, 4, 4] 变换矩阵
    Returns:
        rotation: [B, 3, 3] 旋转矩阵
        translation: [B, 3] 平移向量
    """
    rotation = transform_matrix[:, :3, :3]  # [B, 3, 3]
    translation = transform_matrix[:, :3, 3]  # [B, 3]

    return rotation, translation


def rotation_matrix_loss(pred_rotation, gt_rotation):
    """计算旋转矩阵损失"""
    diff = pred_rotation - gt_rotation
    return torch.mean(torch.sum(diff ** 2, dim=[1, 2]))


def translation_vector_loss(pred_translation, gt_translation):
    """计算平移向量损失"""
    diff = pred_translation - gt_translation
    return torch.mean(torch.sum(diff ** 2, dim=1))


def registration_loss(p_src, transform_pred, transform_gt):
    """
    改进的配准损失函数（保持原接口不变）

    Args:
        p_src: [B, N, 3] 源点云
        transform_pred: [B, 4, 4] 预测的变换矩阵
        transform_gt: [B, 4, 4] 真实的变换矩阵
    Returns:
        total_loss: 总损失
    """

    # 1. 原有的点云变换损失（MSE）
    p_src_pred = transform_points(p_src, transform_pred)
    p_src_gt = transform_points(p_src, transform_gt)
    point_loss = F.mse_loss(p_src_pred, p_src_gt)

    # 2. Chamfer Distance损失
    chamfer_loss = chamfer_distance(p_src_pred, p_src_gt)

    # 3. 分解变换矩阵
    pred_rotation, pred_translation = decompose_transform_matrix(transform_pred)
    gt_rotation, gt_translation = decompose_transform_matrix(transform_gt)

    # 4. 分离的旋转和平移损失
    rotation_loss = rotation_matrix_loss(pred_rotation, gt_rotation)
    translation_loss = translation_vector_loss(pred_translation, gt_translation)

    # 5. 整体变换矩阵损失
    matrix_loss = F.mse_loss(transform_pred, transform_gt)

    # 6. 加权组合（重点强化平移损失）
    total_loss = (
            0.5 * point_loss +  # 原有点云损失
            1.0 * chamfer_loss +  # Chamfer Distance
            1.0 * rotation_loss +  # 旋转损失
            3.0 * translation_loss +  # 平移损失（重点加强）
            0.3 * matrix_loss  # 整体矩阵损失
    )

    return total_loss