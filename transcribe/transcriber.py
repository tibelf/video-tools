import whisperx
import torch
import os
import json
import logging
import numpy as np
from typing import List, Dict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_audio(audio_path: str) -> str:
    """
    预处理音频文件，确保可以被 whisperx 处理
    """
    try:
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"音频文件为空: {audio_path}")
            raise ValueError(f"Audio file is empty: {audio_path}")
        
        logger.info(f"成功加载音频文件: {audio_path}")
        return audio_path
    except Exception as e:
        logger.exception(f"音频文件预处理失败: {e}")
        raise

def transcribe_audio(audio_path: str, max_words_per_segment: int = 50, language: str = 'zh') -> Dict:
    """
    使用 WhisperX 进行音频转录
    
    :param audio_path: 音频文件路径
    :param max_words_per_segment: 每个片段的最大单词数
    :param language: 音频语言
    :return: 包含转录文本和时间戳的字典
    """
    try:
        logger.info(f"开始音频转录: 语言={language}, 最大单词数={max_words_per_segment}")
        
        # 预处理音频
        audio_path = preprocess_audio(audio_path)
        
        # 检查 GPU 可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "float32"
        
        logger.info(f"使用设备: {device}, 计算类型: {compute_type}")
        
        # 加载模型
        logger.info("正在加载 WhisperX 模型...")
        model = whisperx.load_model('large-v2', device, compute_type=compute_type)
        
        # 转录音频
        logger.info("开始音频转录...")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, language=language)
        
        # 对齐时间戳
        logger.info("正在对齐时间戳...")
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, show_progress=False)
        
        # 自然分段处理
        logger.info("处理音频分段...")
        segments = []
        current_segment = {
            'text': '',
            'start': None,
            'end': None,
            'words': []
        }
        
        for segment in result['segments']:
            for word in segment['words']:
                # 确保word字典包含必要的键，并处理可能的异常值
                word_data = {
                    'word': word.get('word', ''),
                    'start': float(word.get('start', 0)),
                    'end': float(word.get('end', 0))
                }
                
                current_segment['words'].append(word_data)
                
                # 如果达到最大单词数，或检测到较长停顿
                if (len(current_segment['words']) >= max_words_per_segment or 
                    (current_segment['words'] and 
                     word_data['start'] - current_segment['words'][-1]['end'] > 1.0)):
                    
                    # 完成当前片段
                    if current_segment['words']:
                        current_segment['text'] = ' '.join([w['word'] for w in current_segment['words']])
                        current_segment['start'] = current_segment['words'][0]['start']
                        current_segment['end'] = current_segment['words'][-1]['end']
                        
                        segments.append(current_segment)
                    
                    # 重置片段
                    current_segment = {
                        'text': '',
                        'start': None,
                        'end': None,
                        'words': [word_data]
                    }
        
        # 处理最后一个片段
        if current_segment['words']:
            current_segment['text'] = ' '.join([w['word'] for w in current_segment['words']])
            current_segment['start'] = current_segment['words'][0]['start']
            current_segment['end'] = current_segment['words'][-1]['end']
            segments.append(current_segment)
        
        logger.info(f"转录完成，共 {len(segments)} 个片段")
        
        return {
            'segments': segments,
            'language': language
        }
    
    except Exception as e:
        logger.exception(f"音频转录过程中发生错误: {e}")
        raise

def save_transcription(transcription: Dict, output_path: str):
    """
    将转录结果保存为 JSON 文件
    
    :param transcription: 转录结果字典
    :param output_path: 输出文件路径
    """
    try:
        logger.info(f"正在保存转录结果到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            # 使用自定义的JSON编码器来处理可能的特殊值
            json.dump(transcription, f, 
                      ensure_ascii=False, 
                      indent=2, 
                      default=_json_serializer)
        logger.info("转录结果保存成功")
    except Exception as e:
        logger.exception(f"保存转录结果失败: {e}")
        raise

def _json_serializer(obj):
    """
    自定义JSON序列化器，处理特殊值
    """
    if isinstance(obj, float):
        # 处理可能的 NaN 或无穷大值
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj

def extract_audio_from_video(video_path: str, output_audio_path: str = None):
    """
    从视频文件中提取音频
    
    :param video_path: 视频文件路径
    :param output_audio_path: 输出音频文件路径，如果未指定，将在视频目录生成
    :return: 音频文件路径
    """
    import subprocess
    
    try:
        logger.info(f"开始从视频文件提取音频: {video_path}")
        
        if not output_audio_path:
            base_name = os.path.splitext(video_path)[0]
            output_audio_path = f"{base_name}.wav"
        
        logger.info(f"输出音频文件: {output_audio_path}")
        
        subprocess.run([
            'ffmpeg', 
            '-i', video_path, 
            '-vn', 
            '-acodec', 'pcm_s16le', 
            '-ar', '16000', 
            '-ac', '1', 
            output_audio_path
        ], check=True)
        
        logger.info("音频提取成功")
        return output_audio_path
    
    except subprocess.CalledProcessError as e:
        logger.exception(f"音频提取失败: {e}")
        raise RuntimeError(f"音频提取失败: {e}")
