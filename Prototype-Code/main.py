# main.py
import cv2
from deepfake_detection.emotion_recognition import detect_emotion
from deepfake_detection.motion_magnification import advanced_motion_magnification
from deepfake_detection.face_swap_detection import analyze_face_swap
from fake_news_detection.sentiment_analysis import analyze_sentiment
from fake_news_detection.tfidf_analysis import analyze_tfidf

def process_video(video_path):
    """Process a video file frame by frame to detect emotion, amplify motion, detect face swaps, and analyze fake news."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    prev_landmarks = None  # Stores landmarks from the previous frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotion in the frame using emotion_recognition module
        emotion, score = detect_emotion(frame)

        # Display the detected emotion and score on the frame
        cv2.putText(frame, f"Emotion: {emotion} ({score:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Detect and amplify motion in the frame using motion_magnification module
        magnified_frame = advanced_motion_magnification(frame, prev_frame)

        # Detect potential face swaps using face_swap_detection module
        frame, prev_landmarks = analyze_face_swap(frame, prev_landmarks)

        # Show the frame with magnified motion, emotion label, and face swap detection
        cv2.imshow('Deepfake Detection: Emotion, Magnification, Face Swap', magnified_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the previous frame for the next iteration
        prev_frame = frame

    cap.release()
    cv2.destroyAllWindows()

def process_fake_news(news_articles):
    """Analyze the news articles for sentiment and TF-IDF relevance."""
    for article in news_articles:
        # Sentiment analysis
        sentiment, score = analyze_sentiment(article['text'])
        print(f"Sentiment: {sentiment} (Score: {score})")

        # TF-IDF Analysis
        tfidf_df = analyze_tfidf([article['text']])
        print("Top TF-IDF terms for this article:")
        print(tfidf_df)

if __name__ == "__main__":
    # Process video for deepfake detection
    video_path = "sample_videos/sample_video.mp4"
    process_video(video_path)

    # Process fake news detection
    news_articles = [
        {"title": "Breaking News: Unbelievable Discovery!", "text": "Scientists have discovered the impossible..."},
        {"title": "Shocking: Government Conspiracy Exposed!", "text": "A government agency has been hiding the truth..."}
    ]
    process_fake_news(news_articles)
