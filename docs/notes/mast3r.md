---
head:
  - - link
    - rel: stylesheet
      href: https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.5.1/katex.min.css
---
# 匹配与立体三维重建

本文介绍匹配与立体三维重建（MASt3R）的基本内容，参考[论文](https://arxiv.org/abs/2406.09756)、[论文](https://arxiv.org/abs/2409.19152)和[中文博客](https://staskaer.github.io/2025/03/03/DUSt3R%E5%92%8CMASt3R%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/#mast3r)，匹配与立体三维重建（MASt3R）事实上是基于密集不受约束的立体三维重建（DUSt3R）进行的，关于此可以阅读我的另一篇[笔记](./dust3r)和[中文博客](https://staskaer.github.io/2025/03/03/DUSt3R%E5%92%8CMASt3R%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/#dust3r)。

## 匹配与立体三维重建

基于密集不受约束的立体三维重建，研究人员针对图像匹配这一特定任务给出了新的方法，即匹配与立体三维重建，这部分主要参考[论文](https://arxiv.org/abs/2406.09756)和[中文博客](https://staskaer.github.io/2025/03/03/DUSt3R%E5%92%8CMASt3R%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB/#mast3r-matching)。

### 匹配头与损失函数

为实现更好的图像匹配，匹配与立体三维重建方法在解码器之后引入了一个新的探测头，以得到每个像素位置的局部特征，用以进行图像匹配。因应图像匹配任务，匹配与立体三维重建方法修改了损失函数，在密集不受约束的立体三维重建方法的损失函数基础上提出了新的匹配损失函数——对比学习损失，旨在造成一张图像的一个局部特征最多与另一张图像的一个局部特征匹配。这里就损失函数作一个简要的介绍，直接查看损失函数公式

$$
L_{match} = -\sum\limits_{(i,j)\in M}log\frac{s(i,j)}{\sum\limits_{k\in P^1}s(k,j)}+log\frac{s(i,j)}{\sum\limits_{k\in P^2}s(k,j)}
$$

其中 $M$ 为真实的匹配对， $s(i,j) = exp(-tD_i^{1T}D_j^2)$ ， $t$ 为温度超参数。很容易发现，这与交叉熵损失有些类似，不同的地方在于加入了温度超参数，而分母不再是所有的类别，而是所有匹配的正样本与负样本。关于对比学习损失，可以参考[知乎文章](https://zhuanlan.zhihu.com/p/506544456)。

匹配与立体三维重建方法还改进了置信度损失部分，将归一化因子统一为真值图的平均深度，而不对于预测图单独设立归一化因子。最后的损失函数即为置信度损失与匹配损失的加权和。

### 快速匹配

得到了局部特征之后，我们需要进行相互匹配，基本的思路是对于每一张图像的每一个像素寻找其在另一张图像上的最近邻，并判断其最近邻在其另一张图像上的最近邻是否为该像素，若是则匹配成功。这样做的复杂度很好，为 $O(H^2W^2)$ ，为了更快的实现匹配，匹配与立体三维重建算法提出了快速匹配方法。

查看相关的源码，代码仓库为[仓库](https://github.com/naver/mast3r)，相关的代码在[文件](https://github.com/naver/mast3r/blob/main/mast3r/fast_nn.py)

```py
def fast_reciprocal_NNs(pts1, pts2, subsample_or_initxy1=8, ret_xy=True, pixel_tol=0, ret_basin=False, device='cuda', **matcher_kw):
    H1, W1, DIM1 = pts1.shape
    H2, W2, DIM2 = pts2.shape
    assert DIM1 == DIM2

    pts1 = pts1.reshape(-1, DIM1)
    pts2 = pts2.reshape(-1, DIM2)

    if isinstance(subsample_or_initxy1, int) and pixel_tol == 0:
        S = subsample_or_initxy1
        y1, x1 = np.mgrid[S // 2:H1:S, S // 2:W1:S].reshape(2, -1)
        max_iter = 10
    else:
        x1, y1 = subsample_or_initxy1
        if isinstance(x1, torch.Tensor):
            x1 = x1.cpu().numpy()
        if isinstance(y1, torch.Tensor):
            y1 = y1.cpu().numpy()
        max_iter = 1

    xy1 = np.int32(np.unique(x1 + W1 * y1))  # make sure there's no doublons
    xy2 = np.full_like(xy1, -1)
    old_xy1 = xy1.copy()
    old_xy2 = xy2.copy()

    if 'dist' in matcher_kw or 'block_size' in matcher_kw \
            or (isinstance(device, str) and device.startswith('cuda')) \
            or (isinstance(device, torch.device) and device.type.startswith('cuda')):
        pts1 = pts1.to(device)
        pts2 = pts2.to(device)
        tree1 = cdistMatcher(pts1, device=device)
        tree2 = cdistMatcher(pts2, device=device)
    else:
        pts1, pts2 = to_numpy((pts1, pts2))
        tree1 = KDTree(pts1)
        tree2 = KDTree(pts2)

    notyet = np.ones(len(xy1), dtype=bool)
    basin = np.full((H1 * W1 + 1,), -1, dtype=np.int32) if ret_basin else None

    niter = 0
    # n_notyet = [len(notyet)]
    while notyet.any():
        _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw))
        if not ret_basin:
            notyet &= (old_xy2 != xy2)  # remove points that have converged

        _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw))
        if ret_basin:
            basin[old_xy1[notyet]] = xy1[notyet]
        notyet &= (old_xy1 != xy1)  # remove points that have converged

        # n_notyet.append(notyet.sum())
        niter += 1
        if niter >= max_iter:
            break

        old_xy2[:] = xy2
        old_xy1[:] = xy1

    # print('notyet_stats:', ' '.join(map(str, (n_notyet+[0]*10)[:max_iter])))

    if pixel_tol > 0:
        # in case we only want to match some specific points
        # and still have some way of checking reciprocity
        old_yx1 = np.unravel_index(old_xy1, (H1, W1))[0].base
        new_yx1 = np.unravel_index(xy1, (H1, W1))[0].base
        dis = np.linalg.norm(old_yx1 - new_yx1, axis=-1)
        converged = dis < pixel_tol
        if not isinstance(subsample_or_initxy1, int):
            xy1 = old_xy1  # replace new points by old ones
    else:
        converged = ~notyet  # converged correspondences

    # keep only unique correspondences, and sort on xy1
    xy1, xy2 = merge_corres(xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy)
    if ret_basin:
        return xy1, xy2, basin
    return xy1, xy2
```

比较复杂，保留其核心内容并作简化（删除一些复杂的条件判断，而保留其最简单情形）即为

```py
def fast_reciprocal_NNs(pts1, pts2, subsample_or_initxy1=8, ret_xy=True, pixel_tol=0):
    H1, W1, DIM1 = pts1.shape
    H2, W2, DIM2 = pts2.shape
    assert DIM1 == DIM2

    pts1 = pts1.reshape(-1, DIM1)
    pts2 = pts2.reshape(-1, DIM2)

    S = subsample_or_initxy1
    y1, x1 = np.mgrid[S // 2:H1:S, S // 2:W1:S].reshape(2, -1)
    max_iter = 10

    xy1 = np.int32(np.unique(x1 + W1 * y1))
    xy2 = np.full_like(xy1, -1)
    old_xy1 = xy1.copy()
    old_xy2 = xy2.copy()

    tree1 = cdistMatcher(pts1)
    tree2 = cdistMatcher(pts2)

    notyet = np.ones(len(xy1), dtype=bool)

    niter = 0
    while notyet.any():
        _, xy2[notyet] = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw))
        if not ret_basin:
            notyet &= (old_xy2 != xy2)

        _, xy1[notyet] = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw))
        notyet &= (old_xy1 != xy1)

        niter += 1
        if niter >= max_iter:
            break

        old_xy2[:] = xy2
        old_xy1[:] = xy1

    converged = ~notyet

    xy1, xy2 = merge_corres(xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy)
    return xy1, xy2
```

很容易看出，基本的方法就是对第一张图像像素进行采样（很稀疏，远小于 $H\times W$ ），然后进行迭代，迭代方法为先从第一张图像的采样点找到第二张图像的最近邻，若与上一次的相同，即视为收敛，剔除出需迭代的集合；对得到的最近邻反向寻找第一张图像的最近邻，并更新这些点，如果与上一次相同，亦视为收敛，剔除出需迭代的集合。算法收敛速度很快，理论保证时间复杂度在 $O(kHW)$ ，其中 $k$ 为采样的像素个数，远小于 $H\times W$ ，因而算法复杂度也远小于原来的 $O(H^2W^2)$ 。

### 粗到细匹配

算法采用了由粗到细匹配的方法，先对图像进行降采样，在降采样结果上进行快速匹配，根据粗匹配的结果选择图像块裁剪，再在裁剪得到的图像块上进行精细的匹配。

## 基于匹配与立体三维重建的运动重建结构

MASt3R SfM 基于 MASt3R matching 的网络展开，为解决运动重建结构（Structure from Motion, SfM）这一经典问题设计。我们不需要训练单独的网络，而是直接使用MASt3R matching的网络。

### 构建场景共视图

为了使输入到完整的MASt3R matching网络的图像对尽可能少，我们先构建场景共视图，即一个图结构，定点为输入的场景，连边表示两个场景存在足够的共视部分，可以用以经过MASt3R matching网络得到点云。共视图的构建步骤为

1. 按编码器输出的标记特征，使用最远点采样方法得到 $N$ 个关键帧，并两两连边
2. 将其余的帧分别连到最近的关键帧，并与其余的普通帧中最近的 $k$ 个相连

### 局部重建

按上面方法得到的场景共视图，对于其每一条边对应的场景图，两两经过网络（事实上上面已经得到了编码器的输出，所以只需要经过解码器），得到对应的至于自己相机坐标系的三维点，注意到一个帧会出现在多个边中，因而会得到多个三维点坐标，我们按照置信度进行加权平均，得到标准点图。通过标准点图，我们可以得到深度图和焦距。定义约束点图为从三维点到相机屏幕的投影的逆映射，这个约束点图有相机内参、相机外参、深度图决定。

### 全局对齐

对于多个输入图对，我们需要进行全局对齐，与密集不受约束的立体三维重建（DUSt3R）不同的是，我们还需要进行捆绑调整。我们使用从粗到细的方法，以密集不受约束的立体三维中的全局对齐方法为粗对齐，不同的地方在于我们的匹配使用的是上面的匹配与立体三维重建（MASt3R-matching）的方法，这会很快地收敛，但得到的点图总还是有些嘈杂，我们需要再进行全局优化，类似于捆绑调整，这里最小化的误差是二维的重投影误差，并使用鲁棒的核函数，比如 $L_2$ 范数

$$
Z^{*}, K^{*}, P^{*}, \sigma^{*} = argmin_{Z,K,P,\sigma}\sum\limits_{c\in M^{n,m}, (n,m)\in e}q_c[\rho(y_c^n - \pi_n(X_c^m)) + \rho(y_c^m - \pi_m(X_c^n))]
$$

很容易注意到这样的优化效果还是有限的。考察一种误差，一帧分别于另外两帧共视，其相邻的两个像素分别于这两帧的两个像素匹配，它们描述的可能是同一个特征点，但没有客观的特征点，所以并不会被关联，这就可能导致了优化的时候这两个像素对应的深度优化结果差距很大，这往往是不合理的，可能导致错误。为解决这个问题，我们假设局部的相对深度是准确的，我们在图像选择一系列锚点，它们均匀并广泛的分布在图像上，将所有的像素与离其最近的锚点关联，其深度替换为锚点的深度与初始时其深度与初始时锚点深度的比例的乘积，而我们要优化的深度图则变化为了锚点组成的深度图。
