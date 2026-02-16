# 🖼️ Imágenes y Assets — DL_Bayesian Paper

> 📊 **Investigación:** Ariel Luján Giamportone
> **ORCID:** [0009-0000-1607-9743](https://orcid.org/0009-0000-1607-9743)

---

## Imágenes del Paper

Figuras incluidas en `DL_Bayesian_Centolla_Paper.tex`:

| Archivo | Figura | Descripción | Sección |
|---------|--------|-------------|---------|
| `fig_arquitectura_sistema.png` | Fig. 1 | Arquitectura del sistema híbrido neuro-bayesiano | §4.1 |
| `fig_dag_causal.png` | Fig. 2 | Grafo Acíclico Dirigido (DAG) de la pesquería | §4.3 |
| `fig_contrafactuales.png` | Fig. 3 | Escenarios contrafactuales (do-calculus) | §4.3 |

## Imágenes para Difusión

Material visual para publicaciones en redes sociales, presentaciones y divulgación:

| Archivo | Concepto | Uso sugerido |
|---------|----------|--------------|
| `fig_gemelo_digital.png` | El Gemelo Digital de la Centolla (Hero) | LinkedIn, ResearchGate, presentaciones |
| `fig_dashboard.png` | Dashboard del Puente de Mando del Futuro | LinkedIn, demos, propuestas |
| `fig_ia_submarina.png` | IA bajo el agua (Concepto visual) | LinkedIn, portadas, banners |

## Uso y licencia

- Todas las imágenes son parte del proyecto DL_Bayesian
- Licencia: MIT (misma que el proyecto)
- Al usar en publicaciones externas, citar:

```
Giamportone, A. L. (2026). Deep Learning y Redes Bayesianas en la Gestión 
de Lithodes santolla. GitHub Repository. 
https://github.com/arielgiamportone/fisheries-sustainability-mlops
```

## Nota técnica para LaTeX

El paper usa `\graphicspath{{images_assets/}}` para referenciar las imágenes.
Compilar desde el directorio `bibliography/`:

```bash
cd bibliography
pdflatex DL_Bayesian_Centolla_Paper.tex
bibtex DL_Bayesian_Centolla_Paper
pdflatex DL_Bayesian_Centolla_Paper.tex
pdflatex DL_Bayesian_Centolla_Paper.tex
```
