# Claude Launcher

Claude Code CLI 的 OpenAI API 代理工具，将 Anthropic API 请求转换为 OpenAI API 格式。

## 安装

```bash
# 克隆项目
git clone https://github.com/myueqf/claude-launch.git
cd claude-launch

# 构建并安装
go build -o claude-launch .
mv claude-launch ~/.local/bin/

# 或者使用构建脚本
chmod +x build.sh
./build.sh
```

## 使用

```bash
claude-launch
```

首次运行会提示：
1. 输入 OpenAI API 端点（如 `https://api.openai.com/v1`）
2. 输入 API 密钥
3. 选择模型

支持参数：
- `--resume` 恢复上次对话
- `--resume=id` 恢复指定对话
- `--continue` 继续上次对话

## 配置

配置文件：`~/.config/claude-launcher/config.json`

```json
{
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-your-key",
  "last_model": ""
}
```

## 功能

- 🔄 Anthropic → OpenAI API 转换
- 📡 完整流式支持（SSE）
- 🛠️ 工具调用转换
- 🤔 思维链（reasoning）支持
- ⚙️ 交互式配置

## 依赖

- Go 1.26+
- Claude Code CLI
