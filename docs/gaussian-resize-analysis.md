# SuperSplat4D 如何避免窗口尺寸变化导致高斯渲染内容被“拉伸”

## 1. 先说结论

这个项目避免“拉伸”的核心，不是某一行单独的 magic code，而是一整条同步链路始终保持一致：

1. DOM/CSS 尺寸变化时，立即拿到新的**实际像素尺寸**。
2. 用这个尺寸更新 canvas 的 drawing buffer，而不是只改 CSS 宽高。
3. 用新的尺寸重建主相机的离屏 `RenderTarget`。
4. 让相机的 `aspectRatio` 基于**当前 render target**自动更新。
5. 让 gsplat shader 使用新的 `matrix_projection` 和新的 `viewport_size` 重新计算每个高斯在屏幕上的椭圆投影。

所以窗口拉伸时，这个项目发生的是：

- 视口变了
- 投影矩阵也跟着变了
- shader 里的屏幕空间高斯尺寸计算也跟着变了

结果是“视野构图变化”或“显示区域变多/变少”，而不是“原画面被二维拉伸”。

你那个项目如果一拉伸窗口，高斯内容也跟着横向/纵向变形，通常说明这条链里至少有一段断了，最常见的是：

- 只改了 canvas 的 CSS 尺寸，没改 drawing buffer 尺寸
- drawing buffer 改了，但相机 aspect 没更新
- 相机 aspect 更新了，但离屏 render target 还是旧尺寸
- 高斯 shader 里仍然在使用旧的 viewport 参数

---

## 2. 这个项目的关键机制

## 2.1 CSS 尺寸和真实渲染尺寸是分开的，但会被同步

这个项目的 canvas 在 CSS 上是铺满容器的：

- [`src/ui/scss/style.scss`](../src/ui/scss/style.scss)
  - `#canvas-container` 是 `width: 100%` 且 `flex-grow: 1`（第 159-165 行）
  - `#canvas` 是 `width: 100%; height: 100%`（第 179-181 行）

这意味着窗口变化时，**CSS 盒子尺寸一定会变**。

但只靠这一步是不够的。因为浏览器里 `<canvas>` 有两套尺寸：

1. CSS 显示尺寸
2. drawing buffer 尺寸，即 `canvas.width / canvas.height`

如果只变第 1 套，不变第 2 套，浏览器就会把旧的像素内容硬拉伸到新的盒子里，这正是你说的“很奇怪”的那种现象。

这个项目在初始化时就先把 drawing buffer 设成容器尺寸乘 `devicePixelRatio`：

- [`src/ui/editor.ts`](../src/ui/editor.ts)
  - 第 422-425 行：
    - 读 `window.devicePixelRatio`
    - `canvas.width = ceil(offsetWidth * pixelRatio)`
    - `canvas.height = ceil(offsetHeight * pixelRatio)`

这只是初始同步。真正关键的是后续 resize 过程。

---

## 2.2 它不用 PlayCanvas 默认自动 resize，而是自己接管

在 `Scene` 构造里：

- [`src/scene.ts`](../src/scene.ts)
  - 第 81 行：`this.app.autoRender = false`
  - 第 83 行：`this.app._allowResize = false`

这里非常关键。

`_allowResize = false` 的含义是：**不走引擎默认那套 “window resize -> app.resizeCanvas()” 的路径**。它自己做更精细的控制。

这么做的原因，是这个项目不是直接把 backbuffer 当最终输出，而是有一套自己的离屏渲染、拷贝、picker、overlay、underlay 流程。默认自动 resize 不足以保证整条链路完全同步。

---

## 2.3 它监听的是容器的实际像素尺寸，不只是 window resize

项目不是简单监听 `window.onresize`，而是直接观察 `#canvas-container`：

- [`src/scene.ts`](../src/scene.ts)
  - 第 97-121 行：`new ResizeObserver(...)`
  - 第 121 行：`observer.observe(document.getElementById('canvas-container'))`

这里有两个重要点：

### 2.3.1 优先使用 `devicePixelContentBoxSize`

在支持的浏览器里：

- 第 101-106 行直接读取 `entry.devicePixelContentBoxSize`

这拿到的是**设备像素级别**的尺寸，不需要自己乘 DPR，最精确。

### 2.3.2 Safari 回退到 `contentBoxSize * devicePixelRatio`

在 Safari 路径里：

- 第 107-114 行

它会用 CSS 盒子尺寸乘 `window.devicePixelRatio` 手动得到像素尺寸。

也就是说，这个项目在 resize 时追踪的不是“逻辑宽高”，而是**真实应该用于渲染的像素宽高**。

得到新尺寸后，它不是立刻改 canvas，而是先记到 `this.canvasResize`，并设置：

- 第 117 行：`this.forceRender = true`

这意味着 resize 和 render 同步发生，避免中间状态错乱。

---

## 2.4 真正改 canvas drawing buffer 的时机在 prerender

在每帧 `onPreRender()`：

- [`src/scene.ts`](../src/scene.ts)
  - 第 360-363 行：
    - `this.canvas.width = this.canvasResize.width`
    - `this.canvas.height = this.canvasResize.height`

这一步非常重要。

它不是只改 CSS，而是直接改 `canvas.width/height`。这会真正改变底层绘制缓冲区分辨率，避免浏览器拿旧帧做二维缩放。

这已经解决了一半问题：**显示层不会把旧内容硬拉伸**。

---

## 2.5 Scene.targetSize 会和当前图形设备尺寸同步

同一个 `onPreRender()` 里还有一层：

- [`src/scene.ts`](../src/scene.ts)
  - 第 367-368 行：
    - `targetSize.width = ceil(graphicsDevice.width / pixelScale)`
    - `targetSize.height = ceil(graphicsDevice.height / pixelScale)`

这里的 `pixelScale` 默认是 1：

- [`src/scene-config.ts`](../src/scene-config.ts)
  - 第 14-17 行

所以正常情况下 `scene.targetSize` 就等于当前设备的实际渲染尺寸。

这意味着后续相机 render target、picker、屏幕空间交互，都不是基于旧尺寸，而是基于这个最新 `targetSize`。

---

## 2.6 主相机会根据 targetSize 重建离屏 RenderTarget

主相机不是一直渲染到一个固定大小的贴图上。

在相机的 `onPreRender()` 里会调用：

- [`src/camera.ts`](../src/camera.ts)
  - 第 624 行：`this.rebuildRenderTargets()`

真正逻辑在：

- [`src/camera.ts`](../src/camera.ts)
  - 第 514-572 行

关键点如下：

### 2.6.1 尺寸来源是 `this.targetSize ?? this.scene.targetSize`

- 第 516 行

平时用 `scene.targetSize`，离屏导出时用手工指定尺寸。这保证了屏幕显示和离屏导出都走同一套机制。

### 2.6.2 如果尺寸没变，就不重建

- 第 519-521 行

只有在宽高或格式改变时才重建，避免额外开销。

### 2.6.3 一旦变了，就销毁旧 RT，创建新 RT

- 第 524-530 行销毁旧 RT
- 第 548-555 行创建新的 `RenderTarget`

这意味着**真正用于高斯渲染的目标纹理尺寸**会随着窗口变化同步更新，不会继续把旧尺寸贴图拉到新窗口里显示。

---

## 2.7 它显式设置 `horizontalFov`，保证 FOV 解释方式和宽高关系一致

在 render target 重建后：

- [`src/camera.ts`](../src/camera.ts)
  - 第 557 行：`this.entity.camera.horizontalFov = width > height`

这行很关键，但容易被忽略。

它的意思不是“改变 FOV 数值”，而是告诉相机：当前视角应该按横向 FOV 还是纵向 FOV 来解释。

窗口横屏和竖屏时，较长边可能互换。如果这里不处理，单一固定解释方式可能会让 framing 或正交尺寸换算不稳定。

这个项目显式按 `width > height` 切换，后面又配合 `fovFactor` 和 `orthoHeight` 做了统一处理：

- [`src/camera.ts`](../src/camera.ts)
  - 第 664-670 行：`fovFactor`
  - 第 601 行：`orthoHeight` 会根据 `horizontalFov` 和当前宽高比调整

这让“以长边定义 FOV，再推回短边 framing”的策略在 resize 后仍然成立。

---

## 2.8 它依赖 PlayCanvas 的 `ASPECT_AUTO`，按当前 RenderTarget 自动更新 aspect

这个项目没有在应用层手工写：

```ts
camera.aspectRatio = width / height;
```

它依赖的是 PlayCanvas camera 的自动纵横比机制。

引擎里：

- [`engine/src/framework/components/camera/component.js`](../engine/src/framework/components/camera/component.js)
  - 第 1223-1227 行：`calculateAspectRatio(rt)` 直接用 `rt.width / rt.height`
  - 第 1238-1240 行：`frameUpdate(rt)` 中，如果 `aspectRatioMode === ASPECT_AUTO`，就把 aspect 更新为当前 RT 的宽高比

renderer 每帧会调用：

- [`engine/src/scene/renderer/renderer.js`](../engine/src/scene/renderer/renderer.js)
  - 第 1102-1103 行：`camera.frameUpdate(renderTarget)`

也就是说，**相机投影的 aspect 不是来自旧窗口尺寸，也不是来自某个初始化常量，而是来自当前 render target 的实时尺寸**。

这是防拉伸的第二个核心。

---

## 2.9 引擎把新的投影矩阵和新的 viewport 尺寸喂给 shader

renderer 设置 camera uniform 时：

- [`engine/src/scene/renderer/renderer.js`](../engine/src/scene/renderer/renderer.js)
  - 第 389 行：`matrix_projection` 被更新
  - 第 441-445 行：`viewport_size = [viewportWidth, viewportHeight, 1/viewportWidth, 1/viewportHeight]`

所以当 render target 尺寸变化后，shader 拿到的也是**新的投影矩阵**和**新的 viewport 尺寸**。

---

## 2.10 gsplat shader 本身就是按“当前投影矩阵 + 当前 viewport”算屏幕椭圆

这部分是最能解释“为什么高斯不会被二维拉伸”的地方。

### 2.10.1 高斯中心先乘当前投影矩阵

- [`engine/src/scene/shader-lib/glsl/chunks/gsplat/vert/gsplatCenter.js`](../engine/src/scene/shader-lib/glsl/chunks/gsplat/vert/gsplatCenter.js)
  - 第 6 行：声明 `uniform mat4 matrix_projection`
  - 第 22 行：`centerProj = matrix_projection * centerView`
  - 第 32 行：`center.projMat00 = matrix_projection[0][0]`

### 2.10.2 高斯椭圆半径和方向再按当前 viewport 计算

- [`engine/src/scene/shader-lib/glsl/chunks/gsplat/vert/gsplatCorner.js`](../engine/src/scene/shader-lib/glsl/chunks/gsplat/vert/gsplatCorner.js)
  - 第 2 行：`uniform vec4 viewport_size`
  - 第 28 行：`float focal = viewport_size.x * center.projMat00`
  - 第 60 行：用 `min(viewport_size.x, viewport_size.y)` 约束 kernel 尺寸
  - 第 70 行：`vec2 c = center.proj.ww * viewport_size.zw`

这说明高斯 quad 的屏幕空间偏移不是“拿一套固定像素半径去画”，而是每帧根据：

- 当前投影矩阵
- 当前 viewport 宽高

重新计算。

所以 resize 后，高斯不是被“把旧像素结果拉伸”，而是**重新投影**。

这就是第三个核心。

---

## 2.11 交互坐标也做了 CSS 尺寸到渲染尺寸的映射

这个项目连 picking 也没有偷懒。

在屏幕点击转 picking 坐标时：

- [`src/camera.ts`](../src/camera.ts)
  - 第 694-696 行：
    - `sx = screenX / target.clientWidth * scene.targetSize.width`
    - `sy = screenY / target.clientHeight * scene.targetSize.height`

这里很关键：输入事件先来自 CSS 坐标，再被映射到真实渲染尺寸。

引擎的 `screenToWorld / worldToScreen` 也使用 `graphicsDevice.clientRect`：

- [`engine/src/framework/components/camera/component.js`](../engine/src/framework/components/camera/component.js)
  - 第 1080-1083 行：`screenToWorld`
  - 第 1093-1096 行：`worldToScreen`

而 `clientRect` 会由 graphics device 更新：

- [`engine/src/platform/graphics/graphics-device.js`](../engine/src/platform/graphics/graphics-device.js)
  - 第 1030-1038 行：`updateClientRect()`

这使交互坐标系、投影坐标系、渲染目标坐标系三者保持一致。

---

## 2.12 overlay / underlay / picker 也跟着主相机同步

这个项目不只有一个主相机输出。还有：

- outline pass
- underlay pass
- picker pass

它们都显式继承主相机的关键投影参数：

- [`src/outline.ts`](../src/outline.ts)
  - 第 109-113 行：复制 `horizontalFov / fov / nearClip / farClip / orthoHeight`
  - 第 116 行：输出到 `this.scene.camera.workRenderTarget`

- [`src/underlay.ts`](../src/underlay.ts)
  - 第 84-88 行：同样复制这些投影参数
  - 第 91 行：也输出到 `workRenderTarget`

而 picker 也会在当前尺寸下重新 `resize`：

- [`src/camera.ts`](../src/camera.ts)
  - 第 786 行：`this.picker.resize(width, height)`

这避免了主渲染不拉伸，但 outline / picking 还在旧分辨率上的错位问题。

---

## 3. 这条“防拉伸”调用链可以总结成什么

可以把它压缩成下面这条链：

1. `#canvas-container` 的 CSS 尺寸改变
2. `ResizeObserver` 拿到新的设备像素尺寸
3. 在 `Scene.onPreRender()` 中写入 `canvas.width / canvas.height`
4. `graphicsDevice.width / height` 跟着更新
5. `scene.targetSize` 根据当前设备尺寸刷新
6. `camera.rebuildRenderTargets()` 用新尺寸重建 RT
7. `camera.frameUpdate(renderTarget)` 用 RT 新尺寸更新 aspect ratio
8. renderer 把新的 `matrix_projection` 和 `viewport_size` 传给 shader
9. gsplat shader 用新的投影和 viewport 重新计算高斯的屏幕空间形状
10. 最终显示的是“重新投影后的结果”，不是“把旧帧图像拉伸”

只要这 10 步都成立，就不会出现你说的那种“窗口一拉，高斯内容也像 2D 图片一样被拉伸”。

---

## 4. 你的项目为什么会拉伸

虽然我还没看你的项目代码，但根据现象，最可能的原因大致按概率排序如下。

## 4.1 只改了 CSS 尺寸，没有改 canvas drawing buffer

典型错误：

```ts
canvas.style.width = '100%';
canvas.style.height = '100%';
```

但是没做：

```ts
canvas.width = canvas.clientWidth * devicePixelRatio;
canvas.height = canvas.clientHeight * devicePixelRatio;
```

这种情况下，浏览器只能把旧的像素内容缩放到新盒子里，视觉上就像高斯画面被整体拉伸。

这是最常见问题。

---

## 4.2 你的主渲染是在离屏纹理里完成的，但离屏纹理没随窗口变化重建

如果你的高斯先画到一个固定大小的 FBO / render target，再 blit 到屏幕：

- 屏幕 canvas 变大了
- 但离屏 RT 还是旧尺寸

那最终也会变成“把旧 RT 内容缩放显示”。

这个项目显式在 [`src/camera.ts`](../src/camera.ts) 第 514-572 行重建 RT，就是为了解决这个问题。

---

## 4.3 相机的 aspect ratio 没更新

即便 canvas 和 RT 尺寸都更新了，如果投影矩阵还按旧的 `width / height` 算：

- NDC 到像素的映射会变
- 但世界到 NDC 的投影没变

最后就会出现横向或纵向比例失真。

这个项目依赖 `camera.frameUpdate(renderTarget)` + `ASPECT_AUTO` 自动修正这件事。

---

## 4.4 shader 里使用了旧的屏幕尺寸 uniform

高斯渲染通常不是纯三角网格。很多实现会在 shader 里显式使用：

- viewport 宽高
- focal / fx / fy
- 1/width, 1/height
- splat scale in pixels

如果你的 shader uniform 仍然是初始化时的旧值，那么哪怕相机 aspect 已经更新，高斯 quad 的屏幕空间尺寸仍然可能按旧屏幕算，结果一样会失真。

这个项目里 `viewport_size` 每帧都会更新，避免了这个问题。

---

## 4.5 输入坐标和渲染坐标系不一致

这个不会直接导致“显示拉伸”，但经常伴随出现：

- 画面看起来没问题
- picking、selection、brush 范围全偏了

这个项目用 `clientWidth/clientHeight -> targetSize.width/height` 的映射统一了两套坐标系。

---

## 5. 如果你要把这个机制迁移到自己的项目，最少要做什么

下面是一份最小可行清单。

## 5.1 必做：统一 resize 流程

窗口或容器 resize 时，做这几步：

1. 读取容器 CSS 尺寸
2. 乘 `devicePixelRatio` 得到实际像素尺寸
3. 更新 `canvas.width / canvas.height`
4. 更新 `graphicsDevice / renderer / swapchain`
5. 重建所有和屏幕相关的 `RenderTarget`
6. 更新相机 aspect ratio
7. 更新 shader 中的 viewport / screenSize / resolution uniform

缺任何一项，都可能导致拉伸或错位。

---

## 5.2 如果你也是离屏渲染，再拷贝到主屏

一定要检查：

- 主相机 RT 是否按窗口变化重建
- 中间工作 RT 是否也同步重建
- final blit 是否只是“同尺寸拷贝”，而不是“把旧 RT 放大贴到屏幕”

---

## 5.3 如果你自己维护相机投影矩阵

每次尺寸变化后至少要更新：

```ts
camera.aspect = width / height;
camera.updateProjectionMatrix();
```

如果是 PlayCanvas 这类引擎，则要确认：

- `aspectRatioMode` 还是自动模式
- 当前 camera 真的绑定到了新的 render target

---

## 5.4 如果你是 gsplat / 3DGS / 2DGS 的自定义 shader

请重点检查以下 uniform 是否在 resize 后同步更新：

- `projectionMatrix`
- `viewProjectionMatrix`
- `viewportSize`
- `screenSize`
- `invScreenSize`
- `fx/fy`
- 任何依赖 `width/height` 的 splat pixel radius 参数

这是最容易漏掉的地方。

---

## 6. 一个可直接参考的实现思路

如果你不想完全照搬 SuperSplat4D，可以把它抽象成下面的伪代码：

```ts
let pendingResize = null;

const resizeObserver = new ResizeObserver((entries) => {
  const entry = entries[0];
  const dpr = window.devicePixelRatio || 1;

  if (entry.devicePixelContentBoxSize) {
    pendingResize = {
      width: entry.devicePixelContentBoxSize[0].inlineSize,
      height: entry.devicePixelContentBoxSize[0].blockSize
    };
  } else {
    pendingResize = {
      width: Math.ceil(entry.contentRect.width * dpr),
      height: Math.ceil(entry.contentRect.height * dpr)
    };
  }

  requestRender();
});

function preRender() {
  if (pendingResize) {
    canvas.width = pendingResize.width;
    canvas.height = pendingResize.height;
    pendingResize = null;
  }

  const width = canvas.width;
  const height = canvas.height;

  rebuildMainRenderTargetIfNeeded(width, height);
  rebuildPickingRenderTargetIfNeeded(width, height);

  camera.aspect = width / height;
  camera.updateProjectionMatrix();

  setUniform("viewport_size", [width, height, 1 / width, 1 / height]);
}
```

如果你的项目是“高斯内容被拉伸”，通常把这套做完整，问题就会消失。

---

## 7. 最值得你直接借鉴的几点

我认为这个项目最值得直接借鉴的不是某个 API，而是这几个设计决策：

1. 不把 resize 只当成 DOM/CSS 事件，而是当成“整个渲染链重配置事件”。
2. 直接观察渲染容器，而不是只监听 `window.resize`。
3. 尺寸统一用设备像素处理，避免 DPR 误差。
4. 主渲染 RT、工作 RT、picker RT 都跟尺寸联动。
5. 相机 aspect 不写死，始终从当前 render target 推导。
6. gsplat shader 每帧使用新的 `matrix_projection` 和 `viewport_size`。
7. 交互坐标也做 CSS 到像素空间的映射。

这 7 条加在一起，才是真正避免“窗口拉伸导致高斯画面被拉伸”的原因。

---

## 8. 针对你项目的排查顺序

如果你现在就要排查你自己的播放器，建议按这个顺序查：

1. resize 后 `canvas.clientWidth/clientHeight` 和 `canvas.width/height` 分别是多少。
2. resize 后主 render target 的 `width/height` 是否真的变了。
3. resize 后相机 projection matrix 是否重新计算了。
4. resize 后 shader 的 `viewport/screenSize` uniform 是否更新了。
5. 最终屏幕显示是不是在把一个旧尺寸 RT 直接拉伸 blit。

只要你把这 5 个点打印出来，基本就能很快定位问题。

---

## 9. 一句话总结

这个项目之所以不会在窗口拉伸时把高斯内容也“拉伸”，本质上是因为它在 resize 后**重新配置了渲染分辨率、render target、相机纵横比和 gsplat shader 的屏幕参数**，所以它渲染的是“新的正确投影结果”，而不是“把旧图像结果缩放显示”。
