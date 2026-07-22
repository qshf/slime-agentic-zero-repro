# Ray Debugging Notes

## 目的

记录 `slime-agentic-zero-repro` 中 Ray 调试的实践、环境依赖、常见问题和当前可用方案。

## 相关文件

- `scripts/ray_debug_demo.py`：已验证可用的 Ray 调试示例脚本
- `mini_slime/train_ray.py`：Ray 同步主循环入口
- `mini_slime/ray/actor_group.py`：Ray actor dispatch 入口
- `1.py`：当前测试时使用的、触发 RemotePdb 的示例脚本

## 环境准备

1. 进入项目根目录：

```bash
cd /Users/qshf/my-project/slime-agentic-zero-repro
```

2. 进入虚拟环境：

```bash
source .venv/bin/activate
```

3. 如果 `.venv` 里没有 pip，先安装 pip：

```bash
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --upgrade pip setuptools wheel
```

4. 安装 Ray dashboard 依赖：

```bash
.venv/bin/python -m pip install "ray[default]"
.venv/bin/python -m pip install debugpy
```

## 启动 Ray 集群

```bash
.venv/bin/ray stop
.venv/bin/ray start --head --include-dashboard true --dashboard-port 8265
```

确认 Ray dashboard 可访问：

```bash
curl --noproxy 127.0.0.1 -I http://127.0.0.1:8265
```

应返回 `HTTP/1.1 200 OK`。

## 常见问题

### 1. `curl 127.0.0.1:8265` 返回 `502 Bad Gateway`

原因：本机环境中存在 `http_proxy` / `https_proxy`，导致本地请求被代理拦截。

解决：

```bash
curl --noproxy 127.0.0.1 -I http://127.0.0.1:8265
```

或者把 `127.0.0.1,localhost` 加到 `NO_PROXY` / `no_proxy`。

### 2. 启动 Ray head 时 dashboard 依赖丢失

如果 `ray start --head --include-dashboard true` 报错：

```
Ray dashboard dependencies failed to install properly: No module named 'aiohttp'
```

说明当前环境安装的是最小 Ray 包，需要安装 `ray[default]`。

### 3. `1.py` 触发的是 RemotePdb 而不是 VS Code Ray Debugger

`1.py` 中使用了：

```python
"RAY_DEBUG": "legacy",
"RAY_DEBUG_POST_MORTEM": "1",
```

这会进入 Ray 的旧式 `RemotePdb` 模式，输出为：

```
RemotePdb session open at localhost:62439, use 'ray debug' to connect...
```

如果要使用 VS Code Ray Debugger，应改成：

```python
"RAY_DEBUG": "1",
"RAY_DEBUG_POST_MORTEM": "1",
```

## VS Code 配置

1. 安装 Ray Debugger 扩展。
2. 添加 Ray cluster：`127.0.0.1:8265`
3. 设置 cluster 的 Local Folder 为：
   `/Users/qshf/my-project/slime-agentic-zero-repro`
4. 运行 Ray 应用后，在 Ray Debugger 面板里点击 paused task attach。

## 推荐调试脚本

当前可用示例：

```bash
.venv/bin/python scripts/ray_debug_demo.py
```

- 不带参数：在 remote task 里触发 `breakpoint()`
- 带参数 `raise`：触发 post-mortem 异常调试

## 调试生产代码的关键点

### `mini_slime/train_ray.py`

这是 Ray 主循环入口，关键点：

- `ray.init(...)`
- `ray.get(rollout_manager.generate.remote(rollout_id))`
- `ray.get(actor_model.async_train(rollout_id, rollout_data))`

### `mini_slime/ray/actor_group.py`

这是训练 actor 的 fan-out 逻辑，关键点：

- `TrainRayActor.remote(...)`
- `async_train(...)` 返回 `ObjectRef` 列表
- `ray.get([a.train.remote(...) for ...])`

## 备注

- 如果你只想验证 Ray 调试环境，本仓库的 `scripts/ray_debug_demo.py` 是最可靠的入口。
- 如果你要调 `1.py`，请先改成 `RAY_DEBUG=1`，否则它会进入旧式 `RemotePdb` 模式。
- `ray debug` 是 legacy 模式的连接方式，和 VS Code Ray Debugger 扩展不是同一个流程。
