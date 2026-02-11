# Deep Learning y Redes Bayesianas en la Gestión de *Lithodes santolla*: Estado del Arte y Oportunidades

La pesquería de centolla patagónica en Argentina presenta una oportunidad única para la aplicación de tecnologías de ciencia de datos, dado su pequeño tamaño de flota, certificación MSC vigente, y la existencia de una sólida base de investigación biológica. Sin embargo, existe una **brecha significativa** entre las metodologías avanzadas de Deep Learning y Redes Bayesianas que se desarrollan globalmente y su aplicación actual en la gestión de esta pesquería. Esta revisión sintetiza el estado del arte global, la literatura específica sobre *L. santolla*, e identifica oportunidades concretas para la implementación de estas tecnologías.

---

## El Deep Learning transforma la gestión pesquera global, pero con sesgo hacia peces óseos

Las aplicaciones de Deep Learning en pesquerías han experimentado un crecimiento exponencial, principalmente en tres áreas: **visión computacional** para identificación y clasificación de especies, **redes recurrentes** para predicción de series temporales, y **modelos de distribución de especies** basados en variables ambientales. Sin embargo, la mayoría de estos desarrollos se concentran en peces óseos, mientras que las aplicaciones específicas para crustáceos permanecen **notablemente subdesarrolladas**.

En **visión computacional**, las arquitecturas YOLO (v3, v4, v7) y Faster R-CNN dominan las aplicaciones para detección de crustáceos. Chen et al. (2023) desarrollaron GMNet-YOLOv4 para clasificación de sexo en cangrejo chino (*Eriocheir sinensis*) con **96.21% de precisión**, mientras que estudios anteriores lograron **98.90% de exactitud** en clasificación de género mediante CNN profundas. Para cangrejo azul, Wang et al. (2018) alcanzaron 97.67% de precisión en detección de articulaciones mediante CNN combinada con k-means. Estas métricas demuestran el potencial para automatizar la determinación de sexo y talla en centolla—parámetros críticos para la gestión "3S" (talla, sexo, temporada) vigente en Argentina.

Las **redes LSTM y arquitecturas Conv-LSTM** para predicción temporal representan la frontera más prometedora. El modelo CATCH desarrollado en Islandia (2024-2025) implementó la primera arquitectura Conv-LSTM para pronóstico espacio-temporal pesquero, logrando RMSE de **4.71×10⁻³** y SSI de **0.955** validado en cinco especies. Crucialmente, no existe ningún modelo LSTM o Transformer publicado para predicción de stocks de cangrejos rey, cangrejo de las nieves, o langosta—un gap directo aplicable a *L. santolla*.

Para **modelos de distribución de especies**, los ensambles de Machine Learning (Boosted Regression Trees, Random Forest, MARS) han sido aplicados exitosamente al cangrejo de las nieves en Alaska. Hardy et al. (2011) desarrollaron el primer modelo predictivo open-access para *Chionoecetes opilio* en el Mar de Bering, integrando variables como extensión de hielo marino y Oscilación Ártica. Modelos BRT recientes (2025) para cangrejo rey rojo de Bristol Bay (*Paralithodes camtschaticus*) utilizan datos dependientes de pesquería para llenar gaps estacionales—una metodología directamente transferible al contexto patagónico.

### Arquitecturas y métricas clave identificadas

| Arquitectura | Aplicación pesquera | Métricas reportadas | Aplicabilidad a centolla |
|-------------|---------------------|---------------------|-------------------------|
| YOLOv4/v7 | Detección y clasificación de cangrejos | mAP 86-95%, Precisión >96% | Alta - clasificación sexo/talla |
| Conv-LSTM | Predicción espacio-temporal de stocks | RMSE 4.71×10⁻³, SSI 0.955 | Alta - pronóstico de capturas |
| BRT/Random Forest | Distribución de especies | AUC "bueno a muy bueno" | Alta - SDM bajo cambio climático |
| CNN-LSTM-ATTN | Predicción de alimentación en acuicultura | N/A | Media - adaptable a variables biológicas |

---

## Las Redes Bayesianas ofrecen marco robusto para incertidumbre y decisiones de manejo

El paradigma bayesiano se ha consolidado como el **estándar de oro** para evaluación de stocks pesqueros debido a su capacidad de cuantificar incertidumbre, incorporar conocimiento previo, y soportar análisis de decisiones basado en riesgo. Los trabajos fundacionales de Punt & Hilborn (1997) y Meyer & Millar (1999) establecieron las bases metodológicas que han evolucionado hacia sofisticados modelos de espacio de estados y redes bayesianas dinámicas.

Los **modelos de espacio de estados bayesianos** separan el error de proceso del error de observación, permitiendo estimaciones más robustas de biomasa y mortalidad. Mäntyniemi et al. (2015) desarrollaron un marco generalizado que aísla procesos biológicos clave y cuenta para correlaciones en supervivencia derivadas de comportamiento gregario y factores ambientales. El modelo WHAM (Woods Hole Assessment Model) representa la implementación más avanzada, vinculando covariables ambientales con reclutamiento y mortalidad natural mediante efectos aleatorios y autocorrelación 2D en supervivencia.

Para **pesquerías de crustáceos específicamente**, Punt et al. (2013) revisaron los modelos integrados estructurados por talla que son ahora estándar para langosta, abulón y cangrejos. El modelo GMACS (General Model for Assessing Crustacean Stocks) de NOAA representa la herramienta más desarrollada, mientras que JABBA (Just Another Bayesian Biomass Assessment) proporciona un marco de producción excedente bayesiano con interfaz R/JAGS. Notablemente, las evaluaciones escocesas de *Cancer pagurus* y *Homarus gammarus* (2016-2019) utilizan análisis de cohorte basado en tallas (LCA) e indicadores basados en longitud (LBI) con puntos de referencia F₀.₁, F₃₅%SpR, y FMAX.

Las **Redes Bayesianas Dinámicas** (DBN) emergen como herramienta poderosa para manejo ecosistémico. Trifonova et al. (2017, 2019) aplicaron DBN con variables latentes y autocorrelación espacial para predecir respuestas de especies a cambios en captura, temperatura y productividad primaria en el Mar del Norte y Golfo de México. Para pesquerías alemanas de platija, Fernandes et al. (2024) desarrollaron una Red de Creencias Bayesiana que integra factores ecológicos, económicos y socioculturales, generando mapas de hotspots de capacidad adaptativa.

### Las Redes Neuronales Bayesianas permanecen inexploradas en pesquerías

Un hallazgo crítico es que las **Bayesian Neural Networks (BNNs)** están **virtualmente ausentes** en la literatura pesquera. Mientras los BNNs se aplican crecientemente en dinámica oceánica (Clare et al., 2022) y predicción climática, no se identificaron aplicaciones ampliamente citadas para evaluación de stocks o manejo pesquero. Esto representa una oportunidad significativa: combinar la capacidad predictiva del Deep Learning con la cuantificación de incertidumbre inherente al enfoque bayesiano podría transformar la toma de decisiones en pesquerías con datos limitados como la centolla.

### Software y herramientas disponibles

| Herramienta | Descripción | Relevancia para centolla |
|------------|-------------|-------------------------|
| JAGS/Stan | Muestreadores MCMC de propósito general | Alta - implementación de modelos bayesianos |
| JABBA | Evaluación bayesiana de biomasa (R/JAGS) | Alta - modelo de producción excedente |
| GMACS | Modelo para stocks de crustáceos | Muy alta - diseñado para crustáceos |
| WHAM | Modelo de espacio de estados Woods Hole | Alta - integración de covariables ambientales |
| TMB | Template Model Builder (C++/R) | Alta - eficiencia computacional |

---

## La investigación sobre *Lithodes santolla* es robusta en biología pero limitada en ciencia de datos

La centolla patagónica cuenta con una **base de conocimiento biológico sustancial** desarrollada principalmente por investigadores del CADIC (Centro Austral de Investigaciones Científicas) e INIDEP, pero carece de aplicaciones modernas de ciencia de datos.

### Parámetros biológicos establecidos

La especie se distribuye desde Uruguay (34°S) hasta Tierra del Fuego (55°S) en el Atlántico, y desde Chiloé (42°S) hasta Cabo de Hornos en el Pacífico, habitando profundidades de **0-700 m** con preferencia térmica de **4-12°C**. El desarrollo larval es lecitotrófico (no alimenticio), comprendiendo 3 etapas zoeales más megalopa, con duración de **19-129 días** dependiendo de la temperatura. Los machos alcanzan madurez gonadal a ~70-75 mm CL y morfométrica a ~105-110 mm CL, requiriendo **7-9 años** para alcanzar talla comercial (≥110 mm CL en Argentina). La longevidad alcanza **hasta 25 años**.

Los trabajos de **Lovrich y colaboradores** (1997, 1999, 2002) en CADIC establecieron las bases sobre potencial reproductivo, crecimiento y apareamiento en Canal Beagle. Calcagno et al. (2003-2005) caracterizaron el desarrollo larval y efectos de temperatura, demostrando que temperaturas **>18°C causan mortalidad larval completa**—un hallazgo con implicaciones críticas bajo escenarios de cambio climático. Varisco et al. (2017-2019) documentaron impactos de bycatch y declive de fecundidad que generan preocupaciones sobre la pesquería actual.

### Primera evaluación cuantitativa de stock (2020)

Canales et al. (2020) condujeron la primera evaluación cuantitativa del Sector Patagónico Central, estableciendo:
- **Biomasa virgen explotable (B₀)**: 20,900 toneladas
- **Biomasa explotable promedio (2016-2019)**: 8,546 toneladas (40.7% de B₀)
- **Mortalidad por pesca objetivo**: F₄₀% = 0.31
- **Mortalidad por pesca actual**: F = 0.204 (por debajo del objetivo)
- **Punto de referencia de biomasa**: B₄₀% como objetivo

La pesquería del Sector Patagónico Central obtuvo **certificación MSC en 2022**, con empresas Bentonicos de Argentina S.A., Crustáceos del Sur S.A., y Centomar S.A. certificadas. Las capturas de la temporada 2023-24 alcanzaron **886 toneladas** (89% de la cuota).

### Brechas de conocimiento identificadas en la literatura específica

La literatura sobre *L. santolla* identifica explícitamente gaps críticos que tecnologías de ciencia de datos podrían abordar:

1. **Patrones de movimiento y conectividad**: Los movimientos dentro y entre stocks permanecen desconocidos. La relación entre poblaciones de Canal Beagle, Sector Central y Sector Sur no está clara.

2. **Reclutamiento**: Los procesos de reclutamiento están pobremente comprendidos, incluyendo requisitos de hábitat para asentamiento juvenil.

3. **Impactos del cambio climático**: Efectos a largo plazo sobre dinámica poblacional proyectados pero no modelados predictivamente.

4. **Limitación espermática**: El alcance del impacto bajo la pesquería actual de solo-machos no está cuantificado.

5. **No recuperación del Canal Beagle**: Después de 22 años de cierre (1992-2014), la población no se ha recuperado. Las causas permanecen inciertas.

---

## Aplicaciones de IA para sostenibilidad pesquera muestran casos de éxito replicables

El panorama global de IA para sostenibilidad pesquera presenta casos de éxito que proveen modelos para implementación en Argentina.

**Global Fishing Watch** utiliza redes neuronales analizando datos AIS combinados con Radar de Apertura Sintética (SAR) para mapear **65,000+ embarcaciones** globalmente, detectando el **75% de embarcaciones** operando sin AIS. Esta tecnología soporta la aplicación de regulaciones en Áreas Marinas Protegidas y ha identificado pesca ilegal sustancial.

**OceanMind** (Microsoft AI for Earth) emplea algoritmos de ML para predecir comportamiento pesquero desde datos de localización de embarcaciones, ofreciendo detección en tiempo casi real de actividades sospechosas a través de Azure cloud. El sistema soporta múltiples agencias de control gubernamentales.

El sistema **AI-RCAS de Corea** implementa YOLOv10 + ByteTrack para análisis de capturas en tiempo real, logrando **74-81% de reconocimiento de especies** a 25 FPS, soportando la aplicación de Capturas Totales Permisibles (TAC).

Para crustáceos específicamente, se han desarrollado sistemas de visión computacional para conteo y medición automatizados usando hardware de bajo consumo (Raspberry Pi 3A+), con prototipos validados para pesquerías del Reino Unido (2023). Sistemas de medición automatizada para camarón norteño (*Pandalus borealis*) han sido validados contra métodos manuales para propósitos de evaluación de stocks.

### Variables ambientales comúnmente integradas

Los modelos exitosos de ML para pesquerías integran consistentemente:
- **Temperatura superficial del mar (SST)**: Variable más comúnmente utilizada; crítica para predicción de caladeros
- **Concentración de clorofila-a**: Indicador de productividad primaria
- **Altura superficial del mar (SSH)**: Patrones de circulación oceánica
- **Índices climáticos**: ENSO, AMO, extensión de hielo marino
- **Batimetría y coordenadas geográficas**: Variables espaciales fundamentales

---

## El contexto argentino presenta digitalización moderada pero ausencia de IA/ML

Argentina ha avanzado en **digitalización básica** de pesquerías pero carece de implementación de inteligencia artificial.

### Sistemas de información implementados

El **SIFIPA** (Sistema Federal de Información de Pesca y Acuicultura), establecido en 2020, funciona como nodo de intercambio de información integrando partes de pesca electrónicos, actas de descarga, transacciones comerciales y registro de artes de pesca. Todas las provincias marítimas (Buenos Aires, Río Negro, Chubut, Santa Cruz, Tierra del Fuego) han firmado convenios de cooperación.

El sistema **GUARDACOSTAS** de Prefectura Naval integra AIS (costero y satelital), posicionamiento VMS, imágenes satelitales y monitoreo mejorado con IA para detección de pesca ilegal. Notablemente, se aplicó la primera multa a embarcación extranjera detectada pescando ilegalmente usando exclusivamente medios electrónicos.

### Brecha tecnológica identificada

| Tecnología | Estado en Argentina | Países líderes |
|-----------|-------------------|----------------|
| Bitácoras electrónicas | Implementado (2019+) | Estándar en UE, Noruega, EE.UU. |
| Monitoreo satelital (VMS) | Operacional | Estándar global |
| Monitoreo electrónico (cámaras) | Solo proyectos piloto (MaRes - UE) | Noruega, UK, EE.UU. ampliamente desplegado |
| IA para evaluación de stocks | **No implementado** | UE (AZTI), Canadá (OnDeck AI) |
| Optimización predictiva de pesca | Uso limitado sector privado | España (Nueva Pescanova - PremIA) |

El **Proyecto MaRes** (financiado por UE) representa la iniciativa de monitoreo electrónico más avanzada de Argentina, con cámaras instaladas en el buque CERES para monitorear líneas espantapájaros, logrando **96% de cumplimiento** en monitoreo de interacciones con aves marinas.

### Capacidad institucional para implementación

El INIDEP mantiene capacidad limitada en ciencia de datos, enfocándose en modelos tradicionales de evaluación de stocks (XSA). CADIC posee fuerte base de investigación pero sin aplicaciones de IA. Comentarios de la industria notan críticamente: "¡Qué lejos del mundo, está quedando la innovación y tecnología argentina en materia pesquera...!"

Las barreras identificadas incluyen:
- **Financiamiento**: Altos costos de inversión inaccesibles para PyMEs
- **Fragmentación institucional**: Jurisdicción dividida entre autoridades nacionales y provinciales
- **Expertise**: Brecha significativa en capacidades de ciencia de datos dentro de instituciones pesqueras
- **Conectividad**: Limitaciones de ancho de banda para reportes en tiempo real desde embarcaciones

---

## Gaps de investigación y oportunidades específicas para Deep Learning y Redes Bayesianas

La síntesis de esta revisión identifica **oportunidades concretas** para aplicación de ciencia de datos en la pesquería de centolla patagónica.

### Aplicaciones de Deep Learning con alta viabilidad

**Clasificación automatizada de sexo y talla mediante CNN**: Los sistemas existentes para cangrejo chino logran >96% de precisión. Dado que la regulación argentina requiere determinación de sexo y verificación de talla mínima (≥110 mm CL), un sistema de visión computacional podría automatizar el proceso de clasificación a bordo, mejorando cumplimiento y reduciendo tiempo de procesamiento.

**Predicción de CPUE y distribución espacial con LSTM/Transformers**: No existe modelo publicado aplicando arquitecturas recurrentes a pesquerías de cangrejos rey. Los datos de VMS, CPUE histórico, y variables ambientales disponibles podrían alimentar modelos Conv-LSTM similares al modelo CATCH islandés.

**Modelos de distribución bajo cambio climático con BRT/Random Forest**: Dada la sensibilidad térmica documentada de *L. santolla* (mortalidad larval >18°C), modelos predictivos bajo escenarios RCP serían valiosos para planificación de manejo a largo plazo.

### Aplicaciones de Redes Bayesianas con alta viabilidad

**Redes Bayesianas Dinámicas para manejo ecosistémico**: Integrar variables de captura, temperatura, productividad primaria, y abundancia de presas (moluscos, ofiuroideos) en un marco probabilístico que capture incertidumbre y permita análisis de escenarios.

**Modelo jerárquico bayesiano para stocks conectados**: Compartir información entre poblaciones de Canal Beagle, Sector Central y Sector Sur bajo estructura jerárquica, aprovechando el enfoque "Robin Hood" para stocks con datos limitados.

**Bayesian Neural Networks para predicción con incertidumbre**: Primera aplicación potencial de BNNs en pesquerías, combinando capacidad predictiva de DL con cuantificación bayesiana de incertidumbre—particularmente valioso para manejo precautorio.

### Datos disponibles para implementación

| Fuente de datos | Contenido | Accesibilidad |
|----------------|-----------|---------------|
| SIFIPA/SIIP | Capturas, descargas, transacciones | Gobierno/usuarios autorizados |
| VMS/SIMPO | Posiciones de embarcaciones, actividad pesquera | Restringido gobierno |
| Programa de Observadores INIDEP | Muestras biológicas, composición de capturas | Restringido/investigación |
| Campañas pre-temporada INIDEP | Biomasa por zona, distribución de tallas | Informes técnicos públicos |
| Variables oceanográficas satelitales | SST, clorofila, SSH | Acceso investigación |

---

## Marco conceptual para integración de DL y Redes Bayesianas en manejo sostenible

La integración propuesta seguiría un **flujo de datos a decisiones**:

1. **Capa de adquisición de datos**: Sistemas de visión (CNN) a bordo para clasificación sexo/talla en tiempo real; integración de datos VMS y ambientales satelitales.

2. **Capa de modelado predictivo**: LSTM/Conv-LSTM para predicción espaciotemporal de CPUE; BRT/RF para distribución bajo escenarios climáticos.

3. **Capa de cuantificación de incertidumbre**: Modelos de espacio de estados bayesianos (JABBA/WHAM) para evaluación de stock; potencialmente BNNs para predicciones con intervalos de credibilidad.

4. **Capa de decisión**: Redes Bayesianas Dinámicas integrando múltiples fuentes de incertidumbre para análisis de opciones de manejo; MSE (Management Strategy Evaluation) para evaluación de reglas de control de captura.

5. **Capa de implementación**: Soporte a TAC zonal; monitoreo de cumplimiento de regulación "3S"; alertas tempranas de cambios en distribución.

---

## Conclusión: Ventana de oportunidad para innovación en pesquería de centolla

La pesquería de centolla patagónica presenta características únicas que la posicionan como **caso piloto ideal** para implementación de tecnologías de ciencia de datos en Argentina:

- **Flota pequeña y controlable** (4-5 buques congeladores autorizados)
- **Certificación MSC vigente** que requiere mejora continua
- **Base robusta de conocimiento biológico** del CADIC
- **Sistema de datos en desarrollo** (SIFIPA) susceptible de integración
- **Cobertura de observadores del 85%** que genera datos biológicos continuos
- **Gaps de conocimiento identificados** directamente abordables con ML

Las metodologías revisadas—CNN para clasificación automatizada, LSTM para predicción temporal, BRT para distribución espacial, modelos bayesianos de espacio de estados para evaluación de stocks, y Redes Bayesianas Dinámicas para manejo ecosistémico—están suficientemente maduras para transferencia tecnológica. La ausencia de BNNs en la literatura pesquera representa además una oportunidad para contribución científica original.

La implementación requerirá **colaboración institucional** entre CADIC (expertise biológico), INIDEP (datos pesqueros), y centros de investigación con capacidad en ciencia de datos, potencialmente con socios internacionales como AZTI (España), NOAA Fisheries, o programas FAO de cooperación técnica. El Proyecto MaRes provee precedente de cooperación exitosa UE-Argentina que podría expandirse hacia aplicaciones de IA.

---

## Referencias bibliográficas clave

### Deep Learning en pesquerías
- Chen et al. (2023). GMNet-YOLOv4 for Chinese mitten crab detection and gender classification. *Computers and Electronics in Agriculture*.
- Hardy et al. (2011). First open-access predictive model for Alaska snow crab. *Integrative and Comparative Biology*.
- Kühn et al. (2024). Machine Learning Applications for Fisheries. *Reviews in Fisheries Science & Aquaculture*.
- CATCH Model (2024-2025). Convolutional-LSTM for spatiotemporal fisheries forecasting. *Biology Methods/Oxford Academic*.

### Redes Bayesianas en pesquerías
- Punt, A.E. & Hilborn, R. (1997). Bayesian stock assessment and decision analysis. *Reviews in Fish Biology and Fisheries* 7:35-63.
- Meyer, R. & Millar, R.B. (1999). BUGS in Bayesian stock assessments. *Canadian Journal of Fisheries and Aquatic Sciences* 56:37-52.
- Mäntyniemi et al. (2015). General state-space framework for Bayesian stock assessment. *ICES Journal of Marine Science* 72:2209-2222.
- Trifonova et al. (2017). Dynamic Bayesian Network for North Sea ecosystem. *ICES Journal of Marine Science* 74:1334-1343.
- Punt et al. (2013). Review of size-structured models for crustaceans and molluscs. *ICES Journal of Marine Science* 70:16-33.
- Winker et al. (2018). JABBA: Bayesian State-Space Surplus Production Model. *Fisheries Research* 204:275-288.

### Literatura sobre *Lithodes santolla*
- Lovrich, G.A. (1997). La pesquería mixta de centollas en Tierra del Fuego.
- Lovrich, G.A. & Vinuesa, J.H. (1999). Reproductive potential of southern king crab. *Scientia Marina* 63:95-105.
- Lovrich et al. (2002). Growth, maturity and mating of male *Lithodes santolla*. *Crustaceana*.
- Calcagno et al. (2003, 2004, 2005). Larval development and temperature effects. *Helgoland Marine Research*.
- Varisco et al. (2017-2019). Bycatch impacts and fecundity decline in *L. santolla*.
- Canales et al. (2020). First quantitative stock assessment for Central Patagonian Sector.
- Lovrich, G.A. & Tapella, F. (2014). Southern King Crabs. In: *King Crabs of the World: Biology and Fisheries Management*. CRC Press.

### IA y sostenibilidad pesquera
- Kroodsma et al. (2018). Tracking the global footprint of fisheries. *Science*.
- Wing, S. & Woodward, C. (2024). Advancing AI in fisheries. *ICES Journal of Marine Science*.
- Rubbens et al. (2023). Machine learning in marine ecology. *ICES Journal of Marine Science*.
- Hodgdon et al. (2022). Global crustacean stock assessment modeling. *Fish and Fisheries*.

### Contexto institucional argentino
- Ley 24.922 - Régimen Federal de Pesca.
- CFP Resolution 19/2008 - Medidas de manejo para pesquería de centolla.
- Ley Provincial 931 (2012) - Regulaciones Tierra del Fuego.
- INIDEP Technical Reports - Informes Técnicos Oficiales (series).
- SIFIPA documentation - Disposición DI-2020-154-APN-SSPYA#MAGYP.