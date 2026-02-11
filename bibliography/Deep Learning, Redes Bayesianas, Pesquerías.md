# **Aplicación de Deep Learning Bayesiano en la Gestión Sostenible de Pesquerías: Un Análisis Integral de la Innovación Tecnológica en el Contexto Global y Argentino**

La intersección entre la inteligencia artificial avanzada y la ciencia pesquera representa uno de los campos más dinámicos de la investigación contemporánea, impulsado por la urgente necesidad de transicionar hacia una "Transformación Azul" que garantice la seguridad alimentaria y la salud de los ecosistemas marinos. En un escenario global donde el 37,7% de las poblaciones de peces se encuentran sobreexplotadas, el despliegue de metodologías de aprendizaje profundo (Deep Learning) basadas en redes bayesianas surge no solo como una mejora técnica, sino como un imperativo ético y estratégico para la gobernanza oceánica. La capacidad de estos modelos para cuantificar la incertidumbre intrínseca de los sistemas biológicos permite a los tomadores de decisiones operar bajo un principio precautorio fundamentado en probabilidades, superando las limitaciones de los modelos deterministas tradicionales que a menudo fallan ante la volatilidad climática y la opacidad de los datos en alta mar.

## **Marco Teórico y Fundamentos del Aprendizaje Profundo Bayesiano**

El aprendizaje profundo convencional ha demostrado una capacidad sin precedentes para identificar patrones en grandes volúmenes de datos, pero carece de un mecanismo robusto para expresar duda. En la gestión de recursos naturales, donde una predicción errónea sobre la biomasa de una especie puede llevar al colapso de una pesquería y a la ruina económica de comunidades costeras, la cuantificación de la incertidumbre es vital. Las Redes Neuronales Bayesianas (BNN) abordan esta carencia asignando distribuciones de probabilidad a cada peso y sesgo de la red, en lugar de valores escalares fijos.

### **Arquitectura de la Inferencia Probabilística**

El núcleo del Deep Learning Bayesiano reside en el Teorema de Bayes, que permite actualizar el conocimiento previo (priors) sobre los parámetros del modelo a medida que se observan nuevos datos. Matemáticamente, la distribución posterior de los pesos \\theta dado el conjunto de datos D se define como:  
Donde p(D | \\theta) representa la verosimilitud (likelihood) y p(D) es la evidencia. Debido a que el cálculo de la evidencia es computacionalmente intratable en redes profundas, se emplean métodos de aproximación como las Cadenas de Markov por Monte Carlo (MCMC) y la Inferencia Variacional (VI). La Inferencia Variacional, en particular mediante algoritmos como ADVI (Automatic Differentiation Variational Inference), permite escalar modelos bayesianos a conjuntos de datos masivos como los generados por el Sistema de Identificación Automática (AIS) de buques, transformando la teoría probabilística en una herramienta operativa para la vigilancia pesquera.

| Característica | Red Neuronal Tradicional | Red Neuronal Bayesiana |
| :---- | :---- | :---- |
| **Parámetros** | Valores fijos (puntos óptimos). | Distribuciones de probabilidad. |
| **Salida** | Predicción puntual única. | Distribución predictiva (media \+ varianza). |
| **Manejo de Incertidumbre** | Nulo o basado en heurísticas. | Intrínseco y fundamentado matemáticamente. |
| **Riesgo de Overfitting** | Alto sin regularización externa. | Bajo, debido al efecto de regularización de los priors. |
| **Costo Computacional** | Bajo (optimización punto a punto). | Alto (requiere muestreo o aproximación funcional). |

Esta capacidad de las BNN para indicar "no lo sé" cuando se enfrentan a datos fuera de la distribución de entrenamiento (incertidumbre epistémica) es lo que las hace indispensables para el monitoreo de pesquerías transzonales y el cumplimiento de acuerdos internacionales como el BBNJ.

## **Contexto Global: La Revolución del Monitoreo y la Inteligencia Artificial**

A nivel internacional, la ciencia pesquera está experimentando una transición desde encuestas independientes de buques de investigación —costosas y de baja frecuencia— hacia sistemas de monitoreo electrónico y análisis de datos de "vessels of opportunity". La democratización de los datos satelitales y el aumento de la capacidad de cómputo han permitido la creación de plataformas como Global Fishing Watch (GFW), que utilizan modelos de Deep Learning para procesar miles de millones de posiciones de buques y clasificar actividades de pesca con una precisión superior al 90%.

### **Integración de Datos AIS y Variables Oceanográficas**

La sostenibilidad pesquera global depende de nuestra capacidad para predecir la abundancia y distribución de las especies en respuesta al cambio climático. Los modelos modernos integran datos de movimiento de buques (AIS) con variables ambientales satelitales para identificar áreas de alta productividad biológica.

1. **AIS como Proxy de Productividad:** Se ha demostrado que los patrones de movimiento de las flotas pesqueras —como los agrupamientos de buques o los cambios en la velocidad de navegación— sirven como indicadores indirectos de la abundancia de peces. Modelos entrenados con datos AIS han logrado precisiones un 18% superiores a los entrenados únicamente con reportes de pesca tradicionales, debido al mayor volumen y continuidad de la información.  
2. **Fusión Multimodal:** La combinación de redes neuronales convolucionales (CNN) con datos de temperatura superficial del mar (SST), clorofila-a y anomalías del nivel del mar permite modelar el hábitat preferencial de especies comerciales clave. En el caso del calamar, arquitecturas como 3DResNet18 han demostrado ser superiores al integrar la dimensión temporal de las variables oceanográficas, reconociendo que la presencia de un recurso hoy depende de la historia térmica de la masa de agua en semanas previas.

### **Desafíos en la Implementación de ML en Ciencias Marinas**

A pesar del progreso, la aplicación de Machine Learning (ML) en pesquerías enfrenta obstáculos críticos relacionados con la robustez y la operabilidad. Los modelos entrenados en una región geográfica a menudo sufren de "domain shift" cuando se aplican a otra, debido a las diferencias en la composición de las especies y las condiciones ambientales. Además, existe una creciente demanda de "explicabilidad" en los modelos de Deep Learning (XAI); los gestores pesqueros necesitan entender *por qué* un modelo predice una disminución en la biomasa antes de imponer restricciones de captura que afectan a miles de trabajadores.

## **El Escenario Regional: Sostenibilidad y Desafíos en el Mar Argentino**

Argentina se sitúa como un actor protagónico en el Atlántico Sur, gestionando una de las áreas marítimas más ricas y complejas del planeta. Con una plataforma continental que se extiende más allá de las 200 millas náuticas, el país enfrenta el reto de administrar recursos transzonales bajo una intensa presión de pesca internacional en la denominada "Milla 201".

### **Estado de las Pesquerías Argentinas y Certificaciones**

A diferencia de la tendencia de sobreexplotación observada en otras áreas del Atlántico Sur (Área 41 de la FAO), la gestión argentina ha logrado mantener niveles de sostenibilidad superiores al promedio mundial. Según el Instituto Nacional de Investigación y Desarrollo Pesquero (INIDEP), solo el 28% de las especies en Argentina se consideran sobreexplotadas, frente al 37,7% global y el 41,2% regional del Área 41\. Este desempeño es el resultado de un marco regulatorio robusto y una colaboración estrecha entre la ciencia y la industria.

| Especie | Volumen Comprometido con Sostenibilidad | Estado de Certificación (2025) |
| :---- | :---- | :---- |
| **Vieira Patagónica** | Alta representatividad histórica. | Certificada desde 2006\. |
| **Centolla** | Recurso de alto valor comercial. | Certificada desde 2022\. |
| **Langostino Argentino** | \> 200,000 toneladas anuales. | Flota costera certificada; altura en evaluación. |
| **Merluza Común** | Principal recurso demersal. | Dos programas de mejoramiento (FIP) activos. |
| **Calamar Argentino** | Alta volatilidad interanual. | Programa de mejoramiento activo y monitoreo por IA. |

Argentina cuenta con más de 645.000 toneladas de captura involucradas en procesos de certificación o mejora, lo que representa el 78% del total de desembarques de especies salvajes. No obstante, pesquerías críticas como la de la merluza negra enfrentan desafíos debido a la falta de planes de manejo aprobados en Áreas Marinas Protegidas (AMP), lo que genera incertidumbre jurídica y operativa para las empresas del sector.

### **Investigación Científica y Tecnología en Argentina**

La soberanía científica sobre el Mar Argentino se ejerce a través de instituciones como el INIDEP y el CONICET, bajo iniciativas estratégicas como Pampa Azul. En 2024 y 2025, se han intensificado las campañas de investigación utilizando tecnología de vanguardia para mapear el talud continental y las zonas adyacentes a la Zona Económica Exclusiva (ZEE).  
Un hito reciente es la expedición "Underwater Oases of Mar Del Plata Canyon", una colaboración con el Schmidt Ocean Institute que permitió la transmisión en vivo desde profundidades de 3.900 metros utilizando el ROV SuBastian. Esta tecnología no solo permite el descubrimiento de nuevas especies, sino que facilita la recolección de ADN ambiental (eDNA), una técnica que, combinada con modelos de Deep Learning para el análisis de secuencias (k-mers), está revolucionando nuestra comprensión de la biodiversidad profunda sin necesidad de muestreos invasivos.

## **Metodología y Desarrollo con Python: Hacia un Sistema de I+D+i**

La elección de Python como lenguaje principal para el desarrollo de proyectos de ciencia de datos en pesquerías no es casual. Su ecosistema de bibliotecas para el aprendizaje profundo y la programación probabilística permite una integración fluida entre la investigación académica (R\&D) y la implementación industrial.

### **Comparativa de Bibliotecas para Deep Learning Bayesiano**

Para desarrollar un sistema de apoyo a la decisión que utilice redes bayesianas, es fundamental seleccionar el backend adecuado según los requisitos de escala y complejidad.

| Biblioteca | Base Computacional | Fortaleza Principal | Aplicación Recomendada |
| :---- | :---- | :---- | :---- |
| **PyMC** | PyTensor (C/JAX/Numba) | Sintaxis intuitiva y modelos jerárquicos. | Evaluación de stock y modelos de biomasa. |
| **Pyro** | PyTorch | Escalabilidad y Deep Learning Bayesiano nativo. | Clasificación de especies en video y análisis AIS. |
| **NumPyro** | JAX | Velocidad extrema en muestreo MCMC/NUTS. | Simulaciones climáticas y proyecciones a largo plazo. |
| **TensorFlow Probability** | TensorFlow | Integración en entornos de producción y capas de Keras. | Sistemas de alerta temprana en tiempo real para flotas. |

En el contexto de un proyecto de I+D+i, el uso de **Pyro** permite definir "Modelos Guía" (variacionales) que aproximan la distribución posterior de los parámetros de una red neuronal, facilitando que el sistema no solo prediga la presencia de un recurso pesquero, sino que también informe sobre la confiabilidad de esa predicción en función de la calidad de los datos de entrada.

### **Implementación Práctica: Predicción de Caladeros de Calamar**

El desarrollo de un modelo para el calamar argentino (*Illex argentinus*) requiere procesar series de tiempo de datos oceanográficos. Utilizando Python, el flujo de trabajo típico incluye la ingesta de datos de satélites (como MODIS-Aqua para clorofila) y modelos de reanálisis (como GLORYS12V1 para temperatura y salinidad).  
Las arquitecturas convolucionales 3D implementadas en **PyTorch** han demostrado ser las más eficaces para capturar la estructura de las "zonas frontales" del océano, donde el encuentro de corrientes (como la de Malvinas y la de Brasil) crea condiciones ideales para la agregación de calamar. Al integrar una capa bayesiana final mediante **PyroSample**, se puede obtener una estimación de la Captura por Unidad de Esfuerzo (CPUE) acompañada de un intervalo de confianza, permitiendo una planificación del esfuerzo pesquero que minimice el consumo de combustible y reduzca la huella de carbono de la flota.

## **La Doble Misión: Capacitación, Divulgación e Innovación**

Un proyecto de ciencia de datos con impacto real debe contemplar la democratización del conocimiento. En Argentina, la brecha tecnológica en el sector pesquero puede cerrarse mediante iniciativas de "Ciencia de Datos Abierta" que utilicen herramientas como GitHub para compartir códigos y metodologías.

### **Estrategias de Divulgación y Educación**

La capacitación no debe limitarse a la formación de expertos en IA; debe extenderse a los pescadores, biólogos de campo y gestores gubernamentales.

1. **Repositorios de Datos Abiertos:** El uso de GitHub para alojar proyectos como los de Fundar (argendata) permite que la comunidad académica y civil acceda a herramientas de ETL para el sector pesquero, fomentando la transparencia y la auditoría ciudadana de los recursos naturales.  
2. **Educación Basada en Tutoriales:** La creación de cuadernos interactivos (Jupyter Notebooks) que guíen al usuario en la construcción de una BNN simple para predecir, por ejemplo, los niveles de oxígeno disuelto a partir de la temperatura, es una herramienta pedagógica poderosa. Estos recursos permiten desmitificar la IA y mostrarla como una aliada de la sostenibilidad.  
3. **Data Journalism Ambiental:** La integración de herramientas de visualización de datos permite traducir la complejidad de los modelos bayesianos en mapas e historias comprensibles para el público general, elevando la conciencia sobre la importancia de la conservación marina y el impacto de la pesca ilegal.

### **El Rol de la I+D+i en el Financiamiento de Proyectos**

Para sustentar el desarrollo tecnológico a largo plazo, es crucial acceder a marcos de financiación nacionales e internacionales. Para el periodo 2025-2026, se destacan convocatorias que priorizan la autonomía estratégica y la digitalización avanzada.

| Programa | Organismo | Foco Temático | Beneficio |
| :---- | :---- | :---- | :---- |
| **Misiones Ciencia e Innovación** | CDTI (España) | Transición energética y soberanía estratégica. | Subvenciones de alta intensidad para consorcios. |
| **Redes Temáticas CYTED** | CYTED (Iberoamérica) | IA aplicada a ODS y desarrollo sostenible. | Financiamiento para cooperación regional (6+ países). |
| **Ecosistemas de Innovación** | CDTI / ANPCyT | Transferencia tecnológica público-privada. | Apoyo a la validación en entorno real. |
| **Programa NEOTEC** | CDTI | Startups de base tecnológica con \< 3 años. | Financiación de nuevos proyectos empresariales. |

La convocatoria de **Ecosistemas de Innovación y Transferencia 2025** es particularmente relevante para proyectos que buscan unir universidades con el sector pesquero industrial, ofreciendo subvenciones para el desarrollo de soluciones que mejoren la trazabilidad y la eficiencia operativa mediante IA. Además, programas como **CYTED** permiten que investigadores argentinos colaboren con pares de España y el resto de Iberoamérica en la creación de redes de IA aplicada a la sostenibilidad energética y alimentaria.

## **Implicaciones para la Sostenibilidad y la Gobernanza**

La aplicación de Deep Learning Bayesiano tiene implicaciones profundas para la gobernanza de las pesquerías, especialmente en un contexto de cambio climático acelerado. Los modelos predictivos permiten pasar de una gestión reactiva —basada en lo que ocurrió el año pasado— a una gestión proactiva que se anticipa a los cambios en la distribución de las especies.

### **Gestión de la Incertidumbre y Cambio Climático**

El aumento de la temperatura del mar y la desoxigenación de las aguas están alterando las rutas migratorias de especies transzonales. Los modelos bayesianos permiten integrar esta "incertidumbre ambiental" en el cálculo de las Capturas Máximas Permisibles (CMP). Si un modelo indica que hay un 40% de probabilidad de que la biomasa caiga por debajo del nivel de seguridad biológica debido a un evento de El Niño, los gestores pueden ajustar las cuotas preventivamente, evitando el colapso que han sufrido otras pesquerías históricas.

### **Autonomía Estratégica y Vigilancia de la Milla 200**

Para Argentina, la soberanía sobre sus recursos pesqueros depende de la capacidad de monitorear su zona económica exclusiva y las áreas adyacentes de manera eficiente. El desarrollo de sistemas nacionales de IA que procesen datos AIS y satelitales reduce la dependencia de proveedores extranjeros y fortalece la posición negociadora del país en foros internacionales como la OMC o el tratado BBNJ. El fortalecimiento de la "Milla 200" no es solo una cuestión de patrullaje físico, sino de superioridad de información y capacidad analítica.

## **Conclusiones y Recomendaciones Futuras**

El proyecto de aplicar Deep Learning basado en redes bayesianas para la sostenibilidad de pesquerías en Argentina se encuentra en el epicentro de la innovación tecnológica y la responsabilidad ecológica. La integración de Python, datos AIS y oceanografía satelital abre un abanico de posibilidades para una gestión más transparente y eficaz.  
A partir del análisis realizado, se desprenden las siguientes recomendaciones estratégicas para el desarrollo del proyecto:

1. **Focalización en Pesquerías de Alta Volatilidad:** El calamar y el langostino, debido a sus ciclos de vida cortos y alta dependencia ambiental, son los candidatos ideales para el despliegue de modelos bayesianos que cuantifiquen la incertidumbre en el reclutamiento anual.  
2. **Adopción de un Enfoque de Ciencia de Datos Abierta:** La publicación de herramientas en GitHub y la participación en redes como CYTED potenciarán el alcance del proyecto, facilitando la capacitación de nuevos profesionales y la validación de las metodologías por parte de la comunidad científica internacional.  
3. **Vinculación con Agendas de I+D+i Internacional:** Es imperativo alinear las propuestas de investigación con las convocatorias de 2025 y 2026 de organismos como el CDTI y la ANPCyT, buscando consorcios público-privados que permitan escalar las soluciones desde el laboratorio hasta la cubierta de los barcos.  
4. **Priorización de la Explicabilidad y la Ética:** El éxito de la IA en la gestión pública depende de la confianza. El desarrollo de modelos que no solo predigan, sino que expliquen sus fundamentos, será esencial para la aceptación de las nuevas regulaciones por parte del sector industrial y los trabajadores de la pesca.

La transformación del sector pesquero argentino mediante la inteligencia artificial bayesiana no es solo una oportunidad de liderazgo regional; es un paso fundamental para asegurar que la inmensa riqueza del Atlántico Sur sea gestionada con la precisión y la prudencia que el siglo XXI demanda.

#### **Fuentes citadas**

1\. Argentina gana protagonismo en la pesca sostenible – Revista Puerto, https://revistapuerto.com.ar/2025/11/argentina-gana-protagonismo-en-la-pesca-sostenible/ 2\. Mapping global fishing activity with machine learning \- Google Blog, https://blog.google/products-and-platforms/products/maps/mapping-global-fishing-activity-machine-learning/ 3\. Advancing High‐Seas Fisheries Governance: Bayesian Models for ..., https://www.researchgate.net/publication/399063187\_Advancing\_High-Seas\_Fisheries\_Governance\_Bayesian\_Models\_for\_Enhancing\_Compliance\_and\_Sustainability\_Under\_the\_BBNJ\_Agreement 4\. Sustainable Marine Ecosystems: Deep Learning for Water Quality Assessment and Forecasting \- UPCommons, https://upcommons.upc.edu/bitstreams/c36dd635-8773-4aef-a81b-961f4326ef4d/download 5\. A neural network model for forecasting fish stock recruitment \- ResearchGate, https://www.researchgate.net/publication/236962722\_A\_neural\_network\_model\_for\_forecasting\_fish\_stock\_recruitment 6\. Making Your Neural Network Say “I Don't Know” — Bayesian NNs using Pyro and PyTorch, https://medium.com/data-science/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd 7\. Tutorial 1: Bayesian Neural Networks with Pyro — UvA DL Notebooks v1.2 documentation, https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial\_notebooks/DL2/Bayesian\_Neural\_Networks/dl2\_bnn\_tut1\_students\_with\_answers.html 8\. Simple Bayesian Neural Network in Pyro \- Kaggle, https://www.kaggle.com/code/carlossouza/simple-bayesian-neural-network-in-pyro 9\. Home — PyMC project website, https://www.pymc.io/ 10\. Global Fishing Watch \- ArcGIS StoryMaps, https://storymaps.arcgis.com/stories/76f8ff21253c427e81e33808918da379 11\. Machine Learning Applications for Fisheries—At Scales from ..., https://repository.library.noaa.gov/view/noaa/68862/noaa\_68862\_DS1.pdf 12\. Learning Fishing Information from AIS Data \- UPCommons, https://upcommons.upc.edu/bitstreams/fbf01f7e-5dc6-413a-8306-583dda5f702e/download 13\. Construction and Comparison of Different Models to Forecast ... \- MDPI, https://www.mdpi.com/2410-3888/10/12/610 14\. Novedades \- Merluza Negra Argentina, https://merluzanegraargentina.org/novedades/ 15\. INIDEP \- Argentina.gob.ar, https://www.argentina.gob.ar/inidep 16\. Instituciones \- Pampa Azul, https://www.pampazul.gob.ar/investigacion-y-desarrollo/instituciones/ 17\. Transmisión en vivo a 3.900 metros de profundidad: en colaboración con el Schmidt Ocean Institute, investigadores del CONICET realizan la expedición “Underwater Oases of Mar Del Plata Canyon: Talud Continental IV”, https://www.conicet.gov.ar/transmision-en-vivo-a-3-900-metros-de-profundidad-en-colaboracion-con-el-schmidt-ocean-institute-investigadores-del-conicet-realizan-la-expedicion-underwater-oases-of-mar-del-plata-canyon/ 18\. Cómo es la tecnología que usa el Conicet para explorar y transmitir desde el mar, https://inteligenciaargentina.ar/nuevas-tecnologias/tecnologia-conicet-para-explorar-mar 19\. Pyro, https://pyro.ai/ 20\. A list of books, events, articles, journals, courses, etc related to Environmental Data Science. \- GitHub, https://github.com/beatrizmilz/Environmental-Data-Science 21\. argendata.fundar \- GitHub, https://github.com/argendatafundar 22\. datos-Fundar/desarrollo\_sector\_pesquero\_acuicola: Este trabajo identifica y analiza las oportunidades más relevantes para la evolución futura del sector pesquero \- GitHub, https://github.com/datos-Fundar/desarrollo\_sector\_pesquero\_acuicola 23\. SohamBera16/bayesian-neural-network-based-prediction-using-Environmental-data \- GitHub, https://github.com/SohamBera16/bayesian-neural-network-based-prediction-using-Environmental-data 24\. Ayudas destinadas a Ecosistemas de Innovación y Transferencia 2025, https://www.ciencia.gob.es/Convocatorias/2025/EcosistemasInnovacion2025.html 25\. Ecosistemas de Innovación y Transferencia 2025 \- CDTI, https://www.cdti.es/ayudas/ecosistemas-de-innovacion-y-transferencia-2025 26\. CYTED Anuncia las Líneas Temáticas de su Convocatoria para 2025, https://www.cyted.org/conteudo.php?idnoticia=1215 27\. CYTED, https://www.cyted.org/ 28\. Estimating fish stock biomass using a Bayesian state-space model: accounting for catchability change due to technological progress \- Frontiers, https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1458257/full 29\. Informe de la pesca y biología de la anchoíta argentina en 2024 \- Pescare, https://pescare.com.ar/informe-de-la-pesca-y-biologia-de-la-anchoita-argentina-en-2024/ 30\. Guía de ayudas CDTI 2025 y novedades de su plan estratégico 2025-2027 | BIK.eus, https://www.bik.eus/noticias/guia-de-ayudas-cdti-2025-y-novedades-de-su-plan-estrategico-2025-2027/ 31\. Ya están disponibles las Bases y Condiciones de las convocatorias de financiamiento para proyectos científicos y tecnológicos | Argentina.gob.ar, https://www.argentina.gob.ar/noticias/ya-estan-disponibles-las-bases-y-condiciones-de-las-convocatorias-de-financiamiento-para