# AI Curtain Virtualizer - Production Grade

A production-ready AI-powered curtain visualization application built with LangChain, Streamlit, and advanced machine learning frameworks.

## Features

### 🚀 Production-Grade Architecture
- **LangChain Integration**: Advanced prompt engineering and model orchestration
- **Async Processing**: Non-blocking image generation with progress tracking
- **Comprehensive Error Handling**: Structured exceptions with retry logic
- **Production Logging**: Structured logging with Loguru
- **Caching**: TTL-based caching for improved performance
- **Configuration Management**: Pydantic-based settings with environment variables

### 🤖 AI Models Supported
- **LangChain + OpenAI**: Enhanced prompt engineering with GPT-4 Vision
- **DALL-E 3**: High-quality image generation
- **Stable Diffusion**: Local GPU-accelerated generation

### 📊 Advanced Features
- **Image Validation**: Comprehensive file type and size validation
- **Color Analysis**: Advanced fabric color extraction and analysis
- **Room Context Analysis**: Intelligent room characteristic detection
- **Progress Tracking**: Real-time generation progress updates
- **Health Checks**: Application monitoring and health endpoints

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai-curtain-virtualizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

4. **Run the application**
```bash
streamlit run main.py
```

### Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

2. **Access the application**
- Open http://localhost:8501 in your browser

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `MODEL_TYPE` | Model to use | `langchain_openai` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CACHE_TTL` | Cache TTL in seconds | `3600` |
| `MAX_RETRIES` | Max retry attempts | `3` |
| `REQUEST_TIMEOUT` | Request timeout in seconds | `30` |

### Model Configuration

```python
# Available models
ModelType.LANGCHAIN_OPENAI  # LangChain + OpenAI (recommended)
ModelType.DALLE             # Direct DALL-E integration
ModelType.STABLE_DIFFUSION  # Local Stable Diffusion
```

## Architecture

### Core Components

1. **ImageProcessor**: Handles image validation, processing, and analysis
2. **ModelFactory**: Manages model instantiation and caching
3. **LangChain Integration**: Advanced prompt engineering and enhancement
4. **Configuration Management**: Pydantic-based settings
5. **Error Handling**: Structured exceptions with detailed error codes

### LangChain Integration

```python
# Enhanced prompt engineering
class LangChainPromptEnhancer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-vision-preview")
        self.chain = self._create_enhancement_chain()
    
    async def enhance_prompt(self, base_prompt, context):
        return await self.chain.ainvoke(context)
```

### Async Processing

```python
# Non-blocking image generation
async def process_images(self, room_photo, fabric_photo):
    # Validate inputs
    self.validate_image(room_photo)
    self.validate_image(fabric_photo)
    
    # Process with retry logic
    for attempt in range(config.max_retries):
        try:
            return await self.model.generate_image(prompt, room_image, fabric_image)
        except Exception as e:
            if attempt == config.max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

## API Reference

### ImageProcessor

```python
class ImageProcessor:
    async def process_images(self, room_photo, fabric_photo) -> Union[Image.Image, str]
    def validate_image(self, image_file) -> None
    def analyze_fabric(self, fabric_image: Image.Image) -> dict
    def analyze_room(self, room_image: Image.Image) -> dict
```

### ModelFactory

```python
class ModelFactory:
    @classmethod
    def get_model(cls, model_type: ModelType) -> BaseModel
```

## Production Deployment

### Docker Deployment

1. **Environment Setup**
```bash
# Create production environment file
cat > .env.prod << EOF
OPENAI_API_KEY=your_api_key_here
LOG_LEVEL=INFO
MODEL_TYPE=langchain_openai
EOF
```

2. **Deploy with Docker Compose**
```bash
docker-compose -f docker-compose.yml up -d
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-curtain-visualizer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-curtain-visualizer
  template:
    metadata:
      labels:
        app: ai-curtain-visualizer
    spec:
      containers:
      - name: app
        image: ai-curtain-visualizer:latest
        ports:
        - containerPort: 8501
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

## Monitoring and Logging

### Structured Logging
- **Console Output**: Colored, formatted logs for development
- **File Logging**: Rotating daily logs with compression
- **Error Tracking**: Separate error logs with extended retention

### Health Checks
- **Application Health**: `/_stcore/health`
- **Model Status**: Automatic model health monitoring
- **Resource Usage**: Memory and CPU monitoring

## Performance Optimization

### Caching Strategy
- **Model Caching**: Singleton pattern for model instances
- **Result Caching**: TTL-based caching for generated images
- **Configuration Caching**: Pydantic model caching

### Async Processing
- **Non-blocking Operations**: All I/O operations are async
- **Progress Tracking**: Real-time progress updates
- **Retry Logic**: Exponential backoff for failed requests

## Security

### Input Validation
- **File Type Validation**: Strict image format checking
- **Size Limits**: Configurable file size restrictions
- **Content Validation**: Image content verification

### API Security
- **Environment Variables**: Secure credential management
- **Request Timeouts**: Configurable timeout limits
- **Rate Limiting**: Built-in retry mechanisms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the logs in the `logs/` directory