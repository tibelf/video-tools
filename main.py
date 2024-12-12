import os
import argparse
import logging
from transcribe.transcriber import extract_audio_from_video, transcribe_audio, save_transcription

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription_main.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='音频转录工具')
    parser.add_argument('input_file', type=str, help='输入视频/音频文件路径')
    parser.add_argument('--output', type=str, help='输出转录文件路径')
    parser.add_argument('--max-words', type=int, default=50, help='每个分段的最大单词数')
    parser.add_argument('--language', type=str, default='zh', help='音频语言')
    
    args = parser.parse_args()
    
    logger.info(f"开始处理文件: {args.input_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        logger.error(f"错误：文件 {args.input_file} 不存在")
        print(f"错误：文件 {args.input_file} 不存在")
        return
    
    # 确定文件类型
    file_extension = os.path.splitext(args.input_file)[1].lower()
    
    try:
        # 处理音频或视频文件
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            # 如果是视频，先提取音频
            logger.info("检测到视频文件，正在提取音频...")
            audio_path = extract_audio_from_video(args.input_file)
        else:
            audio_path = args.input_file
        
        # 设置输出路径
        if args.output:
            output_path = args.output
        else:
            base_name = os.path.splitext(args.input_file)[0]
            output_path = f"{base_name}_transcription.json"
        
        logger.info(f"输出文件路径: {output_path}")
        
        # 执行转录
        transcription = transcribe_audio(
            audio_path, 
            max_words_per_segment=args.max_words, 
            language=args.language
        )
        
        # 保存转录结果
        save_transcription(transcription, output_path)
        
        logger.info(f"转录完成，结果已保存到 {output_path}")
        print(f"转录完成，结果已保存到 {output_path}")
    
    except Exception as e:
        logger.exception(f"转录过程中发生错误: {e}")
        print(f"转录过程中发生错误: {e}")

if __name__ == "__main__":
    main()
