# Torch_Memo

---
```python
import torch
import torch.nn as nn
from einops import rearrange

x = torch.einsum("b h l j, b j d h -> b l d h", x, v)
x = rearrange(x, "b l d h -> b l (d h)")
```
<br/>

---
### einops 패키지 : maxrix 차원 정렬

#### rearrange

```python
pos = repeat(pos, "c h w -> b c h w", b=8)
pos = rearrange(pos, "b c h w-> b c (h w)") 
```

```python
q = rearrange(q, "b l (c h) -> b l c h", c=self.n_hidden)
x = rearrange(x, "b l d h -> b l (d h)")
```
<br/>

#### torch.einsum(b l j, b j d → b l d)  : matrix 곱 ( 3차원 이상 차원 맞춰줘 !!)

```python
x = torch.einsum("b h l j, b j d h -> b l d h", x, v)
```
<br/>

#### numpy 브로드케스팅

<br/>
---
