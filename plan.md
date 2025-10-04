# Plano: Padronização de Modelos de Upscaling (2x e 4x)

> **⚠️ BREAKING CHANGE**: O padrão de upscaling será alterado de 2x para 4x. Usuários que dependem do comportamento anterior precisarão usar `--upscale-scale 2` explicitamente.

## Objetivo
Remover suporte a 3x, padronizar modelos Real-ESRGAN em todos os backends e definir 4x como padrão:
- **2x**: `RealESRGAN_x2plus` (todos backends)
- **4x**: `RealESRGAN_x4plus_anime_6B` (todos backends) - **PADRÃO**

---

## Análise do Estado Atual

### Backends e Modelos Atuais

**1. PyTorch (upscale.py) - Linux/Windows**
- 2x: `RealESRGAN_x2plus.pth` ✅
- 4x: `realesr-general-x4v3.pth` ❌ (precisa trocar)

**2. Core ML (upscale_coreml.py) - macOS**
- 2x: `RealESRGAN_x2plus.mlpackage` ✅
- 4x: `RealESRGAN_x4plus_anime_6B.mlpackage` ✅ (já correto!)

**3. NCNN (upscale_ncnn.py) - macOS legacy**
- 4x: `realesrgan-x4plus-anime` ✅ (equivalente, já correto)

### Suporte a 3x
- **CLI**: `--upscale-scale` já restringe a `choices=[2, 4]` ✅
- **Config**: Valida apenas se é `int`, sem validação de range ❌
- **upscale.py**: Lógica `if scale == 2 else` captura qualquer valor, inclusive 3 ❌

---

## Arquivos a Modificar

### 1. **mangadex_downloader/upscale.py** (PyTorch backend)

**Mudanças:**

**Linha 43-46**: Atualizar `MODEL_HASHES`
```python
MODEL_HASHES = {
    'RealESRGAN_x2plus.pth': '49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb',
    'RealESRGAN_x4plus_anime_6B.pth': 'f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da'
}
```

**Linha 63-78**: Clarificar validação e lógica de modelos
```python
def _init_model(self):
    if self.scale not in [2, 4]:
        raise ValueError(
            f"Unsupported scale factor: {self.scale}. "
            f"PyTorch upscaler supports: 2, 4"
        )

    if self.scale == 2:
        model_name = 'RealESRGAN_x2plus.pth'
        model_scale = 2
    else:  # scale == 4
        model_name = 'RealESRGAN_x4plus_anime_6B.pth'
        model_scale = 4

    # ... resto do código
```

**Linha 127-131**: Atualizar URL de download
```python
if model_name == 'RealESRGAN_x2plus.pth':
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
else:  # RealESRGAN_x4plus_anime_6B.pth
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
```

---

### 2. **mangadex_downloader/config/utils.py**

**Adicionar nova função de validação:**
```python
def validate_upscale_scale(val):
    val = validate_int(val)
    if val not in [2, 4]:
        raise ConfigTypeError(
            f"Upscale scale must be 2 or 4, got {val}"
        )
    return val
```

**Exportar na lista `__all__`:**
```python
__all__ = (
    # ... existing exports
    "validate_upscale_scale",
)
```

---

### 3. **mangadex_downloader/config/config.py**

**Linha 42**: Importar novo validador
```python
from .utils import (
    # ... existing imports
    validate_upscale_scale,
)
```

**Linha 121**: Usar validador específico e alterar padrão para 4
```python
"upscale_scale": (4, validate_upscale_scale),
```

---

### 4. **mangadex_downloader/cli/args_parser.py**

**Linha 263**: Alterar valor padrão de 2 para 4
```python
upscale_group.add_argument(
    "--upscale-scale",
    type=int,
    choices=[2, 4],
    default=config.upscale_scale,
    help="Upscale factor (default: 4)",
)
```

---

### 5. **README.md**

**Linha 52** (aproximadamente): Atualizar descrição
```markdown
- **Optional image upscaling (2x or 4x) using Real-ESRGAN with hardware acceleration.**
```

**Linhas 58-78** (seção "Image Upscaling"): Expandir informação
```markdown
### Backends
- **macOS (Apple Silicon):** Uses Core ML with Neural Engine
  - 2x: RealESRGAN_x2plus
  - 4x: RealESRGAN_x4plus_anime_6B (optimized for anime/manga, default)
- **Linux / Windows:** Uses PyTorch with CUDA/CPU
  - 2x: RealESRGAN_x2plus
  - 4x: RealESRGAN_x4plus_anime_6B (optimized for anime/manga, default)

### How to Use
mangadex-dl "URL" --upscale                     # 4x upscaling (default)
mangadex-dl "URL" --upscale --upscale-scale 2   # 2x upscaling
mangadex-dl "URL" --upscale --upscale-scale 4   # 4x upscaling (explicit)
```

---

### 6. **CLAUDE.md**

**Seção "PyTorch Backend (Linux/Windows)"**: Atualizar modelos
```markdown
**Key features:**
- **Models**: Supports 2x and 4x upscaling
  - **2x**: `RealESRGAN_x2plus.pth`
    - URL: `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth`
    - SHA256: `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb`
  - **4x**: `RealESRGAN_x4plus_anime_6B.pth`
    - URL: `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth`
    - SHA256: `f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da`
```

**Seção "Model URLs and Hashes"**: Atualizar PyTorch
```markdown
### PyTorch (Linux/Windows)
- **2x**: SHA256 `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb`
- **4x**: SHA256 `f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da`
```

---

### 7. **docs/changelog.md**

**Adicionar nova versão no topo:**
```markdown
## v3.2.1-upscale

### Improvements

- **Standardized 4x Upscaling Model**: Switched PyTorch backend from `realesr-general-x4v3.pth` to `RealESRGAN_x4plus_anime_6B.pth`
  - All backends now use the anime-optimized 4x model for consistent quality
  - New SHA256: `f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da`
- **4x is now the default scale**: Changed default upscale factor from 2x to 4x for better quality
  - Use `--upscale-scale 2` to explicitly use 2x upscaling if needed
- **Enhanced Validation**: Added strict scale validation (2 or 4 only) at config level
- **Removed 3x Support**: Eliminated legacy 3x upscaling code paths for clarity

### Breaking Changes

- **Default upscale scale changed from 2x to 4x**: Existing users relying on 2x default behavior will need to explicitly set `--upscale-scale 2`
- Users who manually downloaded `realesr-general-x4v3.pth` will need to re-download the new model (automatic on first use)
```

---

## Resumo das Mudanças

### Arquivos Modificados
1. ✅ `mangadex_downloader/upscale.py` - Trocar modelo 4x e adicionar validação
2. ✅ `mangadex_downloader/config/utils.py` - Adicionar `validate_upscale_scale`
3. ✅ `mangadex_downloader/config/config.py` - Usar novo validador e alterar padrão para 4
4. ✅ `mangadex_downloader/cli/args_parser.py` - Alterar help text para refletir default=4
5. ✅ `README.md` - Atualizar documentação de uso
6. ✅ `CLAUDE.md` - Atualizar hashes e URLs
7. ✅ `docs/changelog.md` - Documentar mudanças

### Arquivos que NÃO precisam modificação
- ✅ `upscale_coreml.py` - Já usa modelo correto
- ✅ `upscale_ncnn.py` - Já usa modelo correto

---

## Testes Necessários

Após implementação:
1. ✅ Reinstalar pacote: `uv pip install --upgrade ".[optional]"`
2. ✅ Testar padrão (4x): `mangadex-dl "URL" --upscale` (deve usar 4x)
3. ✅ Testar 2x explícito: `mangadex-dl "URL" --upscale --upscale-scale 2`
4. ✅ Testar 4x explícito: `mangadex-dl "URL" --upscale --upscale-scale 4`
5. ✅ Verificar rejeição de 3x: `mangadex-dl "URL" --upscale --upscale-scale 3` (deve falhar)
6. ✅ Verificar download automático do novo modelo 4x (`RealESRGAN_x4plus_anime_6B.pth`)
7. ✅ Verificar validação de hash SHA256 (deve ser `f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da`)
8. ✅ Verificar que `--help` mostra "default: 4" na opção `--upscale-scale`

---

## Considerações de Desempenho

**4x vs 2x:**
- 4x produz imagens com **16x mais pixels** que a original (vs 4x no caso de 2x)
- Processamento 4x é ~2-3x mais lento que 2x
- Arquivos finais serão ~2-4x maiores (dependendo da compressão)
- **Recomendação**: 4x é ideal para leitura em telas grandes e arquivamento; 2x é suficiente para dispositivos móveis

**Impacto:**
- Usuários com hardware limitado podem preferir 2x
- Download de modelos: 4x anime model (~17.9MB) vs generic 4x model (~64MB) - **economia de ~46MB**

---

## Benefícios

1. **Consistência**: Todos backends usam modelos anime-otimizados
2. **Qualidade Superior por Padrão**: 4x upscaling padrão oferece melhor qualidade para mangá/anime
3. **Modelo Otimizado**: `RealESRGAN_x4plus_anime_6B` é superior para mangá/anime vs modelo genérico
4. **Segurança**: Validação estrita de valores de scale (2 ou 4 apenas)
5. **Clareza**: Código mais explícito, sem else genérico
6. **UX Melhorada**: Usuários obtêm melhor qualidade sem precisar especificar escala
7. **Download Menor**: Modelo anime 4x é ~73% menor que o modelo genérico anterior
