"""Tokenisation shared by the retrievers and the answer selector."""
import re

_TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset("""a an and are as at be by for from has have in is it its of on or that the this to was were
which with who what when where whose why how not no than then there these those their his her he she they
them we you your our i me my do does did can could would should may might will shall into about after before
during over under most more less least very also such each other any all both either neither""".split())


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOP]
