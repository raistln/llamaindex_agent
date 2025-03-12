import os
import subprocess
import sys
import time
import platform
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

# Verificar y manejar dependencias
try:
    from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool, ToolMetadata
    from llama_index.llms.ollama import Ollama
except ImportError:
    print("🔄 Instalando dependencias necesarias...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-index", "llama-index-llms-ollama", "requests"])
    print("✅ Dependencias instaladas correctamente.")
    from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import FunctionTool, ToolMetadata
    from llama_index.llms.ollama import Ollama


# Clase principal que maneja todo el sistema multiagente
class MultiAgentSystem:
    def __init__(self, model_name="phi", data_dir="./data", temperatura=0.7, timeout=120.0, verbose=True):
        """
        Inicializa el sistema multiagente.
        
        Args:
            model_name: Nombre del modelo de Ollama a utilizar
            data_dir: Directorio donde se almacenarán los documentos de conocimiento
            temperatura: Temperatura para el modelo (mayor valor = más creativo)
            timeout: Tiempo máximo de espera para respuestas
            verbose: Si se deben mostrar logs detallados
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.temperatura = temperatura
        self.timeout = timeout
        self.verbose = verbose
        
        self.llm = None
        self.indice = None
        self.recuperador = None
        self.coordinador = None
        
        # Crear directorio de datos si no existe
        Path(data_dir).mkdir(exist_ok=True)
        
        print(f"🚀 Iniciando Sistema Multiagente con modelo '{model_name}'")
    
    def instalar_ollama(self) -> bool:
        """Instala Ollama en el sistema si no está presente"""
        sistema = platform.system().lower()
        
        print("🔄 Instalando Ollama...")
        
        try:
            if sistema == "linux":
                # Instalación en Linux
                print("Detectado sistema Linux. Instalando Ollama...")
                subprocess.run(
                    "curl -fsSL https://ollama.com/install.sh | sh",
                    shell=True,
                    check=True
                )
                
            elif sistema == "darwin":  # macOS
                # Instalación en macOS
                print("Detectado sistema macOS. Instalando Ollama...")
                
                # Verificar si Homebrew está instalado
                try:
                    subprocess.run(["brew", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    # Instalar con Homebrew
                    subprocess.run(["brew", "install", "ollama"], check=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    # Si no hay Homebrew, usar el script oficial
                    subprocess.run(
                        "curl -fsSL https://ollama.com/install.sh | sh",
                        shell=True,
                        check=True
                    )
                    
            elif sistema == "windows":
                # Instalación en Windows
                print("Detectado sistema Windows. Descargando instalador de Ollama...")
                
                # URL del instalador de Windows
                url = "https://ollama.com/download/windows"
                
                # Descargar el instalador
                installer_path = os.path.expanduser("~/Downloads/ollama-installer.exe")
                response = requests.get(url)
                with open(installer_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Instalador descargado en {installer_path}")
                print("Ejecutando instalador. Por favor complete la instalación cuando aparezca el asistente...")
                
                # Ejecutar el instalador
                subprocess.run([installer_path], check=True)
                
                # Esperar a que el usuario complete la instalación
                print("Esperando a que se complete la instalación...")
                time.sleep(10)  # Dar tiempo para que el instalador se inicie
                
                # Verificar si el servicio de Ollama está en ejecución
                max_attempts = 5
                for attempt in range(max_attempts):
                    try:
                        subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                        print("✅ Ollama instalado correctamente en Windows")
                        break
                    except FileNotFoundError:
                        if attempt < max_attempts - 1:
                            print(f"Esperando a que Ollama esté disponible... ({attempt+1}/{max_attempts})")
                            time.sleep(10)
                        else:
                            print("⚠️  La instalación puede estar en progreso. Si este script falla:")
                            print("1. Complete la instalación de Ollama manualmente")
                            print("2. Asegúrese de que Ollama esté en ejecución")
                            print("3. Vuelva a ejecutar este script")
            else:
                print(f"❌ Sistema operativo no compatible: {sistema}")
                print("Por favor, instale Ollama manualmente desde: https://ollama.ai/")
                return False
            
            # Verificar si la instalación fue exitosa
            try:
                # Dar tiempo a que el servicio se inicie
                time.sleep(3)
                
                # Intentar ejecutar comando de Ollama
                result = subprocess.run(
                    ["ollama", "list"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    print("✅ Ollama instalado correctamente y funcionando")
                    return True
                else:
                    print(f"⚠️  Ollama instalado pero presenta errores: {result.stderr}")
                    return False
                    
            except FileNotFoundError:
                print("⚠️  Ollama instalado pero no se encuentra en el PATH del sistema")
                print("Recomendación: Reinicie su terminal o computadora y vuelva a intentarlo")
                return False
                
        except Exception as e:
            print(f"❌ Error durante la instalación de Ollama: {str(e)}")
            print("Por favor, instale Ollama manualmente desde: https://ollama.ai/")
            return False
    
    def verificar_ollama(self) -> bool:
        """Verifica si Ollama está instalado y disponible. Si no, intenta instalarlo"""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                print("❌ Ollama está instalado pero no funciona correctamente.")
                print(f"Error: {result.stderr}")
                
                # Intentar reparar o reinstalar
                return self.instalar_ollama()
                
            print("✅ Ollama está instalado y funcionando correctamente")
            return True
            
        except FileNotFoundError:
            print("❌ Ollama no está instalado en este sistema.")
            return self.instalar_ollama()
    
    def verificar_modelo(self) -> bool:
        """Verifica si el modelo solicitado está disponible y lo descarga si es necesario"""
        try:
            # Verificar si el modelo ya está descargado
            result = subprocess.run(
                ["ollama", "list"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                encoding="utf-8",  # Forzar codificación UTF-8
                check=False
            )
            
            if self.model_name in result.stdout:
                print(f"✅ Modelo '{self.model_name}' ya está disponible")
                return True
            else:
                print(f"🔄 Descargando modelo '{self.model_name}'...")
                download = subprocess.run(
                    ["ollama", "pull", self.model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",  # Forzar codificación UTF-8
                    check=False
                )
                
                if download.returncode != 0:
                    print(f"❌ Error al descargar el modelo: {download.stderr}")
                    return False
                
                print(f"✅ Modelo '{self.model_name}' descargado correctamente")
                return True
                
        except Exception as e:
            print(f"❌ Error al verificar/descargar el modelo: {str(e)}")
            return False
    
    def crear_documentos_ejemplo(self):
        """Crea documentos de ejemplo si no existen"""
        if not os.listdir(self.data_dir):
            print("📄 Creando archivos de ejemplo...")
            
            examples = {
                "python.txt": """
                Python es un lenguaje de programación interpretado de alto nivel creado por Guido van Rossum en 1991.
                Es conocido por su filosofía de diseño que enfatiza la legibilidad del código con su notable uso de espacios en blanco.
                Python soporta múltiples paradigmas de programación, incluyendo programación estructurada, orientada a objetos y funcional.
                Es ampliamente utilizado en ciencia de datos, inteligencia artificial, desarrollo web y automatización.
                """,
                
                "agents.txt": """
                Los agentes de IA son sistemas de software que pueden percibir su entorno, tomar decisiones y actuar para lograr objetivos específicos.
                Un agente utiliza modelos de lenguaje (LLMs) como su "cerebro" para procesar información y tomar decisiones.
                Los multiagentes son sistemas donde varios agentes especializados colaboran para resolver tareas complejas.
                Cada agente puede tener un rol específico: búsqueda de información, análisis, creatividad, etc.
                El flujo típico incluye: recibir una solicitud, planificar, ejecutar acciones mediante herramientas, y entregar resultados.
                """,
                
                "rag.txt": """
                RAG (Retrieval Augmented Generation) es una técnica que combina la recuperación de información con la generación de texto.
                En un sistema RAG, cuando se recibe una consulta, primero se buscan documentos o fragmentos relevantes en una base de conocimiento.
                Luego, estos fragmentos se proporcionan como contexto a un modelo de lenguaje para generar una respuesta informada.
                RAG mejora la precisión de las respuestas al proporcionar información específica y actualizada.
                También ayuda a reducir las "alucinaciones" o la generación de información incorrecta por parte de los modelos.
                """,
                
                "llms.txt": """
                Los Large Language Models (LLMs) son modelos de inteligencia artificial entrenados con enormes cantidades de texto.
                Estos modelos pueden generar texto, traducir idiomas, escribir diferentes tipos de contenido creativo y responder preguntas.
                Ejemplos populares incluyen GPT, LLaMA, Mistral, Phi y Claude.
                Los LLMs locales son modelos que se pueden ejecutar en tu propio ordenador o servidor, sin necesidad de conexión a la nube.
                Ollama es una herramienta que permite ejecutar LLMs de forma local con una configuración mínima.
                """
            }
            
            for filename, content in examples.items():
                with open(os.path.join(self.data_dir, filename), "w", encoding="utf-8") as f:
                    f.write(content)
            
            print("✅ Archivos de ejemplo creados correctamente")
    
    def configurar_modelo(self):
        """Configura el modelo LLM de Ollama"""
        try:
            self.llm = Ollama(
                model=self.model_name,
                temperature=self.temperatura,
                request_timeout=self.timeout
            )
            Settings.llm = self.llm
            print("✅ Modelo configurado correctamente")
            return True
        except Exception as e:
            print(f"❌ Error al configurar el modelo: {str(e)}")
            return False
    
    def crear_indice(self):
        """Crea un índice vectorial con los documentos del directorio de datos"""
        try:
            documentos = SimpleDirectoryReader(self.data_dir).load_data()
            self.indice = VectorStoreIndex.from_documents(documentos)
            self.recuperador = self.indice.as_retriever(similarity_top_k=2)
            print(f"✅ Índice creado con {len(documentos)} documentos")
            return True
        except Exception as e:
            print(f"❌ Error al crear el índice: {str(e)}")
            
            # Crear un recuperador simulado si falla la creación del índice real
            def recuperador_simulado(query):
                return [{"text": "No se pudo crear un índice real, esta es una respuesta simulada."}]
            
            self.recuperador = type('obj', (object,), {
                'retrieve': recuperador_simulado
            })
            return False
    
    def definir_herramientas(self):
        """Define las herramientas que utilizarán los agentes"""
        # Herramienta de búsqueda de conocimiento
        conocimiento_tool = FunctionTool.from_defaults(
            name="buscador_conocimiento",
            fn=self.buscar_conocimiento,
            description="Busca información en la base de conocimientos interna"
        )
        
        # Herramienta de calculadora
        calculadora_tool = FunctionTool.from_defaults(
            name="calculadora",
            fn=self.calculadora,
            description="Realiza operaciones matemáticas básicas como suma, resta, multiplicación y división"
        )
        
        # Herramienta de búsqueda web simulada
        web_tool = FunctionTool.from_defaults(
            name="buscador_web",
            fn=self.buscar_web_simulada,
            description="Busca información en la web (simulado)"
        )
        
        print("✅ Herramientas base creadas correctamente")
        return conocimiento_tool, calculadora_tool, web_tool
    
    def crear_agentes(self, conocimiento_tool, calculadora_tool, web_tool):
        """Crea los agentes especializados y el coordinador"""
        try:
            # Agente investigador (especializado en búsqueda de información)
            agente_investigador = ReActAgent.from_tools(
                [conocimiento_tool, web_tool],
                llm=self.llm,
                verbose=self.verbose,
                system_prompt="""
                Eres un agente investigador especializado en buscar y recuperar información.
                Tu objetivo es encontrar datos precisos y relevantes utilizando las herramientas disponibles.
                Primero intenta buscar en la base de conocimientos interna, y si no encuentras información,
                utiliza el buscador web. Presenta la información de manera clara y organizada.
                """
            )
            
            # Agente analista (especializado en matemáticas y análisis)
            agente_analista = ReActAgent.from_tools(
                [calculadora_tool],
                llm=self.llm,
                verbose=self.verbose,
                system_prompt="""
                Eres un agente analista especializado en matemáticas y análisis de datos.
                Tu objetivo es realizar cálculos precisos y analizar información numérica.
                Utiliza la calculadora para realizar operaciones matemáticas cuando sea necesario.
                Explica tus razonamientos paso a paso para que sean fáciles de entender.
                """
            )
            
            print("✅ Agentes especializados creados correctamente")
            
            # Convertir los agentes en herramientas para el coordinador
            investigador_tool = FunctionTool.from_defaults(
                name="investigador",
                fn=lambda query: agente_investigador.chat(query).response,
                description="Consulta al agente investigador para buscar información"
            )
            
            analista_tool = FunctionTool.from_defaults(
                name="analista",
                fn=lambda query: agente_analista.chat(query).response,
                description="Consulta al agente analista para realizar análisis matemáticos"
            )
            
            # Crear el agente coordinador
            self.coordinador = ReActAgent.from_tools(
                [investigador_tool, analista_tool],
                llm=self.llm,
                verbose=self.verbose,
                system_prompt="""
                Eres un agente coordinador que gestiona un equipo de agentes especializados.
                Tu objetivo es resolver las consultas del usuario delegando tareas específicas a los agentes adecuados:
                
                - Para búsqueda de información y consultas generales, utiliza al agente "investigador"
                - Para cálculos matemáticos y análisis numéricos, utiliza al agente "analista"
                
                Analiza cada consulta para determinar qué agente debería manejarla, o si se requiere la colaboración
                de varios agentes. Organiza y sintetiza las respuestas de los agentes para proporcionar una
                respuesta final coherente y completa al usuario.
                """
            )
            
            print("✅ Sistema multiagente creado correctamente")
            return True
        except Exception as e:
            print(f"❌ Error al crear los agentes: {str(e)}")
            return False
    
    def buscar_conocimiento(self, query: str) -> str:
        """
        Busca información en la base de conocimientos.
        
        Args:
            query: La consulta o pregunta del usuario
            
        Returns:
            Información relevante encontrada en la base de conocimientos
        """
        try:
            resultados = self.recuperador.retrieve(query)
            if not resultados:
                return "No se encontró información relevante en la base de conocimientos."
            
            textos = [node.text for node in resultados]
            return "\n\n".join(textos)
        except Exception as e:
            return f"Error al buscar en la base de conocimientos: {str(e)}"
    
    def calculadora(self, operacion: str, a: float, b: float) -> str:
        """
        Realiza operaciones matemáticas básicas.
        
        Args:
            operacion: Tipo de operación (suma, resta, multiplicacion, division, potencia, raiz)
            a: Primer número
            b: Segundo número
            
        Returns:
            El resultado de la operación
        """
        try:
            if operacion.lower() == "suma":
                return f"{a} + {b} = {a + b}"
            elif operacion.lower() == "resta":
                return f"{a} - {b} = {a - b}"
            elif operacion.lower() == "multiplicacion":
                return f"{a} * {b} = {a * b}"
            elif operacion.lower() == "division":
                if b == 0:
                    return "Error: No se puede dividir por cero"
                return f"{a} / {b} = {a / b}"
            elif operacion.lower() == "potencia":
                return f"{a} ^ {b} = {a ** b}"
            elif operacion.lower() == "raiz":
                if a < 0 and b % 2 == 0:
                    return "Error: No se puede calcular raíz par de un número negativo"
                return f"Raíz {b} de {a} = {a ** (1/b)}"
            elif operacion.lower() == "modulo" or operacion.lower() == "resto":
                return f"{a} % {b} = {a % b}"
            else:
                return (f"Operación '{operacion}' no reconocida. Opciones válidas: " 
                        f"suma, resta, multiplicacion, division, potencia, raiz, modulo")
        except Exception as e:
            return f"Error en la calculadora: {str(e)}"
    
    def buscar_web_simulada(self, query: str) -> str:
        """
        Simula una búsqueda en la web (para evitar dependencias externas).
        
        Args:
            query: La consulta o búsqueda
            
        Returns:
            Resultados simulados de la búsqueda
        """
        base_conocimiento = {
            "python": "Python es un lenguaje de programación versátil usado en ciencia de datos, web y automatización.",
            "javascript": "JavaScript es un lenguaje de programación para desarrollo web que permite crear páginas interactivas.",
            "llama index": "LlamaIndex es una biblioteca para crear aplicaciones con LLMs y fuentes de datos personalizadas.",
            "agentes": "Los agentes de IA son sistemas autónomos que perciben, deciden y actúan para lograr objetivos.",
            "rag": "RAG (Retrieval Augmented Generation) combina recuperación de datos con generación de texto.",
            "ollama": "Ollama es una herramienta para ejecutar modelos de lenguaje de forma local sin necesidad de servicios en la nube.",
            "phi": "Phi es un modelo de lenguaje pequeño desarrollado por Microsoft, diseñado para funcionar en ordenadores personales con recursos limitados.",
            "llama": "LLaMA (Large Language Model Meta AI) es una familia de modelos de lenguaje desarrollados por Meta AI (Facebook).",
            "mistral": "Mistral es una familia de modelos de lenguaje desarrollados por Mistral AI, con versiones optimizadas para diferentes tamaños y casos de uso.",
            "gemma": "Gemma es un modelo de lenguaje desarrollado por Google, optimizado para ejecutarse en computadoras con recursos limitados.",
            "cpu": "La CPU (Unidad Central de Procesamiento) es el 'cerebro' del ordenador que ejecuta instrucciones.",
            "gpu": "La GPU (Unidad de Procesamiento Gráfico) está optimizada para operaciones paralelas y es ideal para IA.",
            "multiagente": "Un sistema multiagente es una red de agentes de IA que colaboran para resolver problemas complejos."
        }
        
        resultados = []
        for keyword, info in base_conocimiento.items():
            if keyword.lower() in query.lower():
                resultados.append(info)
        
        if not resultados:
            return f"No se encontró información sobre: {query}"
        
        return "\n".join(resultados)
    
    def responder(self, consulta: str) -> str:
        """
        Procesa una consulta del usuario utilizando el sistema multiagente.
        
        Args:
            consulta: La consulta o pregunta del usuario
            
        Returns:
            Respuesta generada por el sistema multiagente
        """
        if self.coordinador is None:
            return "Error: El sistema multiagente no está inicializado correctamente."
        
        if self.verbose:
            print(f"\n📝 Consulta recibida: {consulta}")
            print("🤖 Procesando con el sistema multiagente...")
        
        try:
            start_time = time.time()
            respuesta = self.coordinador.chat(consulta)
            elapsed_time = time.time() - start_time
            
            if self.verbose:
                print(f"⏱️ Tiempo de respuesta: {elapsed_time:.2f} segundos")
            
            return respuesta.response
        except Exception as e:
            return f"Error al procesar la consulta: {str(e)}"
    
    def inicializar(self) -> bool:
        # Comprobar Ollama
        # if not self.verificar_ollama():
        #     return False
        
        # Comprobar y descargar modelo si es necesario
        if not self.verificar_modelo():
            return False
        
        # Configurar modelo
        if not self.configurar_modelo():
            return False
        
        # Crear documentos de ejemplo
        self.crear_documentos_ejemplo()
        
        # Crear índice vectorial
        self.crear_indice()
        
        # Definir herramientas
        conocimiento_tool, calculadora_tool, web_tool = self.definir_herramientas()
        
        # Crear agentes
        if not self.crear_agentes(conocimiento_tool, calculadora_tool, web_tool):
            return False
        
        print("\n" + "="*80)
        print("🚀 Sistema multiagente inicializado correctamente")
        print("="*80)
        return True
    
    def ejecutar_ejemplos(self):
        """Ejecuta ejemplos de consultas para demostrar el funcionamiento"""
        print("\n" + "="*80)
        print("🧪 Ejecutando ejemplos de consultas")
        print("="*80)
        
        ejemplos = [
            "¿Qué es Python y para qué se utiliza?",
            "Necesito calcular 23.5 * 17.8",
            "¿Qué es RAG y puedes calcular cuánto es 15 al cuadrado?"
        ]
        
        for i, consulta in enumerate(ejemplos, 1):
            print(f"\nEjemplo {i}: {consulta}")
            respuesta = self.responder(consulta)
            print(f"\nRespuesta: {respuesta}")
            time.sleep(1)  # Breve pausa entre ejemplos
        
        print("\n" + "="*80)
        print("✅ Ejemplos completados")
        print("="*80)
    
    def iniciar_interfaz_consola(self):
        """Inicia una interfaz interactiva por consola"""
        print("\n🤖 Asistente Multiagente LlamaIndex")
        print("Escribe 'salir' para terminar")
        print("Escribe 'ayuda' para ver comandos disponibles")
        
        while True:
            try:
                consulta = input("\n🧠 Consulta: ")
                
                # Comandos especiales
                if consulta.lower() in ["salir", "exit", "quit", "q"]:
                    print("¡Hasta luego!")
                    break
                elif consulta.lower() in ["ayuda", "help", "h", "?"]:
                    print("\n📖 Comandos disponibles:")
                    print("- salir/exit/quit: Terminar la aplicación")
                    print("- ayuda/help: Mostrar esta ayuda")
                    print("- modelos: Listar modelos disponibles")
                    print("- cambiar [modelo]: Cambiar al modelo especificado")
                    print("- status: Mostrar estado del sistema")
                    continue
                elif consulta.lower() == "modelos":
                    try:
                        result = subprocess.run(
                            ["ollama", "list"], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            text=True,
                            check=False
                        )
                        print("\n📋 Modelos disponibles en Ollama:")
                        print(result.stdout)
                    except Exception as e:
                        print(f"Error al listar modelos: {str(e)}")
                    continue
                elif consulta.lower().startswith("cambiar "):
                    nuevo_modelo = consulta.split(" ", 1)[1].strip()
                    print(f"🔄 Cambiando al modelo '{nuevo_modelo}'...")
                    
                    self.model_name = nuevo_modelo
                    if self.verificar_modelo() and self.configurar_modelo():
                        print(f"✅ Modelo cambiado a '{nuevo_modelo}' correctamente")
                        # Recrear los agentes con el nuevo modelo
                        conocimiento_tool, calculadora_tool, web_tool = self.definir_herramientas()
                        self.crear_agentes(conocimiento_tool, calculadora_tool, web_tool)
                    else:
                        print(f"❌ No se pudo cambiar al modelo '{nuevo_modelo}'")
                    continue
                elif consulta.lower() == "status":
                    print("\n📊 Estado del sistema:")
                    print(f"- Modelo actual: {self.model_name}")
                    print(f"- Temperatura: {self.temperatura}")
                    print(f"- Directorio de datos: {self.data_dir}")
                    print(f"- Sistema multiagente activo: {'Sí' if self.coordinador else 'No'}")
                    continue
                    
                # Procesar consulta normal
                respuesta = self.responder(consulta)
                print(f"\n🤖 Respuesta: {respuesta}")
                
            except KeyboardInterrupt:
                print("\n¡Hasta luego!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


# Función principal
def main():
    # Configurar modelos disponibles
    modelos_disponibles = ["phi", "llama3", "mistral", "gemma"]
    modelo_predeterminado = "phi3"
    
    # Elegir modelo
    if len(sys.argv) > 1 and sys.argv[1] in modelos_disponibles:
        modelo = sys.argv[1]
    else:
        print(f"\n📋 Modelos disponibles: {', '.join(modelos_disponibles)}")
        print(f"Usando modelo predeterminado: {modelo_predeterminado}")
        modelo = modelo_predeterminado
    
    # Inicializar el sistema
    sistema = MultiAgentSystem(
        model_name=modelo,
        temperatura=0.7,  # Valor predeterminado para equilibrar creatividad y precisión
        verbose=True      # Mostrar logs detallados
    )
    
    # Inicializar el sistema multiagente
    if sistema.inicializar():
        # Ejecutar ejemplos demostrativos
        if len(sys.argv) > 2 and sys.argv[2].lower() == "--ejemplos":
            sistema.ejecutar_ejemplos()
        
        # Iniciar la interfaz de consola interactiva
        sistema.iniciar_interfaz_consola()
    else:
        print("❌ No se pudo inicializar el sistema multiagente correctamente.")
        print("Por favor, verifica la instalación de Ollama y los modelos.")
        sys.exit(1)

# Punto de entrada cuando se ejecuta como script
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Programa terminado por el usuario.")
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        sys.exit(1)