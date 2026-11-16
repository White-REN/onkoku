from pathlib import Path
import json
import re
import sys
import urllib.request

import paths

ROOT = paths.RESOURCE_DIR

CONFIG_FILE = paths.CONFIG_FILE


def load_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{CONFIG_FILE}"
        )

    with CONFIG_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)

def build_prompt(text, previous_ja=None, previous_zh=None):
    """带上一条日语/译文作为上下文。

    断句是按 20 字切的，一句日语经常被拆成两条。逐条翻译时，后一条不知道
    前一条讲什么，人称、称谓、术语会飘（实测 8 处里 3 处不一致：店主/老板娘、
    家臣/武家、器具/器械）。带上一条做锚点就能对齐，而且**不额外耗时**——
    实测 44 条 16.7 秒 vs 不带的 16.1 秒。

    也试过「多条一批 + 编号」：格式能 100% 对齐，但反而更慢（18.5s，因为
    标号本身也要生成），还会丢句子（「已经决定」整句没翻）、自己加说话人
    前缀（「佐纪：」）、把「勝先生」原样留在中文里。所以没采用。
    """
    lines = [
        "将下面的日语字幕翻译成自然、准确、符合影视对白习惯的简体中文。",
        "字幕来自连续对白，请参考上一条的用语、人称和称谓，保持前后一致。",
        "只输出这一条的中文译文，不要解释，不要添加引号，不要输出日文。",
        "",
    ]

    if previous_ja:
        lines.append(f"上一条日语：{previous_ja}")
        lines.append(f"上一条译文：{previous_zh or '（无）'}")
        lines.append("")

    lines.append(f"日语：{text}")

    return "\n".join(lines)


def translate_text(text, previous_ja=None, previous_zh=None):

    config = load_config()

    translation_config = (
        config["translation"]
    )

    engine = translation_config.get(
        "engine",
        "local"
    )

    api_url = translation_config.get(
        "api_url",
        "http://127.0.0.1:8080/v1/chat/completions"
    )

    api_key = translation_config.get(
        "api_key",
        ""
    )

    api_model = translation_config.get(
        "api_model",
        ""
    )

    model = translation_config.get(
        "model",
        "HY-MT2-7B-Q4_K_M.gguf"
    )

    prompt = build_prompt(
        text,
        previous_ja,
        previous_zh,
    )

    if engine == "local":
        request_model = model

    elif engine == "api":

        if not api_model:
            raise RuntimeError(
                "API 模式没有设置 api_model。"
            )

        request_model = api_model

    else:

        raise RuntimeError(
            f"未知的翻译引擎：{engine}"
        )


    body = {
        "model": request_model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.1,
        "top_p": 0.6,
        "max_tokens": 256,
        "stream": False,
    }


    data = json.dumps(
        body,
        ensure_ascii=False
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }

    if api_key:
        headers["Authorization"] = (
            f"Bearer {api_key}"
        )

    request = urllib.request.Request(
        api_url,
        data=data,
        headers=headers,
        method="POST",
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

    except Exception as e:

        raise RuntimeError(
            f"翻译 API 请求失败：{e}"
        )

    try:

        translated = (
            result["choices"][0]["message"]["content"]
            .strip()
        )

    except (
        KeyError,
        IndexError,
        TypeError
    ):

        raise RuntimeError(
            f"翻译 API 返回格式异常：\n{result}"
        )

    if not translated:

        raise RuntimeError(
            "翻译 API 返回了空译文"
        )

    return translated


def get_output_file(input_file):

    input_file = Path(input_file)

    if input_file.name.endswith(".ja.srt"):
        filename = (
            input_file.name[:-7]
            + ".zh.srt"
        )
    else:
        filename = (
            input_file.stem
            + ".zh.srt"
        )

    return (
        paths.TEMP_DIR
        / filename
    )


def parse_blocks(text):
    """把 SRT 切成 [序号, 时间轴, 文本]，保持原顺序。"""
    blocks = []

    for block in re.split(
        r"\r?\n\s*\r?\n",
        text.strip(),
    ):
        lines = block.splitlines()

        if len(lines) < 2:
            continue

        blocks.append([
            lines[0].strip(),
            lines[1].strip(),
            "\n".join(lines[2:]).strip(),
        ])

    return blocks


def resume_prefix(blocks, output_file):
    """上次翻到哪儿了，返回已确认可复用的输出块。

    只按「序号 + 时间轴」对齐，不比对正文（正文已经是中文了）。对不上就
    说明这份字幕不是同一段视频的，直接当没有进度，别把别的片子接着翻。
    """
    try:
        done = parse_blocks(output_file.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError):
        return []

    if not done:
        return []

    for index, (number, timeline, body) in enumerate(done):

        if index >= len(blocks):
            return []

        if number != blocks[index][0] or timeline != blocks[index][1]:
            print("已有的中文 SRT 与这次输入对不上，重新开始翻译")
            return []

        # 最后一条只写了一半（进程被杀），丢掉它重新翻
        if not body and blocks[index][2]:
            return done[:index]

    return done


def translate_srt(input_file):

    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(
            f"找不到字幕文件：{input_file}"
        )

    output_file = get_output_file(input_file)

    print("=" * 50)
    print("Subtitle Translator")
    print("=" * 50)
    print(f"输入：{input_file}")
    print(f"输出：{output_file}")
    print()

    text = input_file.read_text(
        encoding="utf-8-sig"
    )

    blocks = parse_blocks(text)

    total = len(blocks)

    # 断点续传：翻一条写一条，进程被杀 / 断电之后重跑能接着往下走
    existing = resume_prefix(blocks, output_file)

    done = len(existing)

    # 续传时，上一句的译文要从已有的输出里接着用，否则续上的第一条会丢上下文
    previous_zh = existing[-1][2] if existing else None

    if done:
        print(
            f"发现上次的进度：{done}/{total} 条已完成，接着翻"
        )
        print()

    if done >= total:
        print("中文字幕已经完整，不需要重翻")
        return output_file

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 先按已确认的前缀重写一遍，顺便丢掉可能写坏的尾条
    with output_file.open("w", encoding="utf-8") as handle:

        for number, timeline, body in existing:
            handle.write(
                f"{number}\n{timeline}\n{body}\n\n"
            )

        for index in range(done, total):

            number, timeline, subtitle_text = blocks[index]

            if not subtitle_text:
                handle.write(f"{number}\n{timeline}\n\n")
                handle.flush()
                continue

            print(
                f"[{index + 1}/{total}] "
                f"{subtitle_text}"
            )

            previous_ja = (
                blocks[index - 1][2]
                if index > 0
                else None
            )

            translated = translate_text(
                subtitle_text,
                previous_ja,
                previous_zh,
            )

            previous_zh = translated

            print(
                f"    → {translated}"
            )

            print(
                f"[TRANSLATE_PROGRESS] {index + 1}/{total}",
                flush=True
            )

            handle.write(
                f"{number}\n{timeline}\n{translated}\n\n"
            )
            handle.flush()

    print()
    print("=" * 50)
    print("翻译完成")
    print("=" * 50)
    print(f"中文 SRT：{output_file}")

    return output_file


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("用法：")
        print(
            'python translate.py "日语字幕.ja.srt"'
        )
        sys.exit(1)

    translate_srt(
        sys.argv[1]
    )