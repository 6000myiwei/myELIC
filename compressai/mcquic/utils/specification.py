from dataclasses import dataclass
from typing import List, Union


@dataclass
class CodeSize:
    """Latent code specification.
           Code in this paper is of shape: `[[1, m, h, w], [1, m, h, w] ... ]`
                                                            `↑ total length = L`

    Args:
        heights (List[int]): Latent height for each stage.
        widths (List[int]): Latent width for each stage.
        k (List[int]): [k1, k2, ...], codewords amount for each stage.
        m (int): M, multi-codebook amount.
    """
    m: int
    heights: List[int]
    widths: List[int]
    k: List[int]

    def __str__(self) -> str:
        sequence = ", ".join(f"[{w}x{h}, {k}]" for h, w, k in zip(self.heights, self.widths, self.k))
        return f"""
        {self.m} code-groups: {sequence}"""
    
if __name__ == '__main__':
    c = CodeSize(m=16, heights=[16,8,4], widths=[16,8,4], k=[8192, 2048, 512])
    print(c)