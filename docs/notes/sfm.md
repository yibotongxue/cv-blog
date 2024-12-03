---
head:
  - - link
    - rel: stylesheet
      href: https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.5.1/katex.min.css
---
# 运动中的结构（Structure from Motion）

我们在[这篇文章](./3d-vision.md)讲了相机标定和三角测量，其中相机标定是从一个相机在不同的地方、不同的角度拍摄的照片上的点对恢复相机的参数，三角测量是从已知参数的相机和照片恢复三维世界的信息，这里我们介绍的 Structure from Motion 则是从一系列不同的相机拍摄的照片及点对中恢复出这些相机的参数和三维信息。

## 基本的问题

假设我们有 $m$ 个相机拍摄的 $m$ 张照片，并知道每张照片上有 $n$ 个点对应的分别的 $n$ 个相同的世界物体，我们要求解得就是这 $m$ 个相机的参数和 $n$ 个世界物体的世界坐标。根据我们对相机成像过程的分析（可以见[这篇文章](./camera-coordinate.md)），我们假设第 $i$ 张照片上的第 $j$ 个点坐标为 $x_{ij}$ ，对应的世界物体的世界坐标为 $X_j$ ，而相机参数矩阵为 $A_i$ ，则有这样的等式成立：

$$
x_{ij} = A_i X_j
$$

而我们已知的就是 $x_ij$ ，共有 $mn$ 个坐标，每个坐标有两个维度（除去齐次坐标），事实上我们可以得到 $2mn$ 个方程。而我们要求解的是 $A_i$ 和 $X_j$ ，其中 $A_i$ 是一个 $3\times 4$ 的矩阵，共有 $12m$ 个未知量，而 $X_j$ 有三个维度（除去齐次坐标），共有 $3n$ 个未知量。

## 歧义性问题

不出意料的，我们想仅仅从一系列不同的相机拍摄的照片及点对恢复出相机的参数和三维信息，会有歧义性的问题。