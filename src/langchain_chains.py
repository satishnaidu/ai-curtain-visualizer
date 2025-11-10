from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict, Any
from .config import config

class PromptEnhancement(BaseModel):
    """Structured output for prompt enhancement"""
    enhanced_prompt: str = Field(description="Enhanced prompt for image generation")
    style_elements: str = Field(description="Key style elements identified")
    technical_specs: str = Field(description="Technical specifications for generation")

class LangChainPromptEnhancer:
    """Production-grade prompt enhancement using LangChain"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.langchain_model_name,
            temperature=config.langchain_temperature,
            max_tokens=config.langchain_max_tokens,
            api_key=config.openai_api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=PromptEnhancement)
        self.chain = self._create_enhancement_chain()
    
    def _create_enhancement_chain(self):
        """Create the prompt enhancement chain"""
        
        system_template = SystemMessagePromptTemplate.from_template(
            "You are an expert interior designer and AI prompt engineer. "
            "Your task is to enhance prompts for generating photorealistic curtain visualizations. "
            "Focus on technical accuracy, aesthetic appeal, and realistic implementation."
        )
        
        human_template = HumanMessagePromptTemplate.from_template(
            "Enhance this curtain visualization prompt:\n\n"
            "Base prompt: {base_prompt}\n"
            "Room context: {room_context}\n"
            "Style preferences: {style_preferences}\n\n"
            "Create a detailed, professional prompt that will generate a high-quality, "
            "photorealistic curtain visualization. Include specific details about:\n"
            "- Fabric draping and texture\n"
            "- Lighting and shadows\n"
            "- Room integration\n"
            "- Professional photography style\n\n"
            "{format_instructions}"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            system_template,
            human_template
        ])
        
        return (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {**x, "format_instructions": self.parser.get_format_instructions()})
            | prompt_template
            | self.llm
            | self.parser
        )
    
    async def enhance_prompt(self, base_prompt: str, room_context: Dict[str, Any], 
                            style_preferences: str = "modern elegant") -> PromptEnhancement:
        """Enhance a base prompt with contextual information"""
        
        input_data = {
            "base_prompt": base_prompt,
            "room_context": str(room_context),
            "style_preferences": style_preferences
        }
        
        result = await self.chain.ainvoke(input_data)
        return result

class LangChainColorAnalyzer:
    """Advanced color analysis using LangChain"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-vision-preview",
            temperature=0.3,
            api_key=config.openai_api_key
        )
        self.chain = self._create_color_analysis_chain()
    
    def _create_color_analysis_chain(self):
        """Create color analysis chain"""
        
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a professional color analyst and interior designer. "
                "Analyze fabric colors and provide detailed color descriptions "
                "suitable for interior design applications."
            ),
            HumanMessagePromptTemplate.from_template(
                "Analyze the dominant colors in this fabric sample and provide:\n"
                "1. Primary color palette (3-5 main colors)\n"
                "2. Color temperature (warm/cool/neutral)\n"
                "3. Saturation level (vibrant/muted/subtle)\n"
                "4. Best complementary room colors\n"
                "5. Interior design style compatibility\n\n"
                "Fabric description: {fabric_description}\n"
                "Provide a concise but comprehensive analysis."
            )
        ])
        
        return (
            prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    async def analyze_colors(self, fabric_description: str) -> str:
        """Analyze fabric colors"""
        result = await self.chain.ainvoke({"fabric_description": fabric_description})
        return result