use opencv::{
    core::{Mat, Scalar},
    imgproc::{cvt_color, COLOR_BGR2GRAY, rectangle},
    objdetect::{CascadeClassifier, CSVMDetector},
    prelude::*,
    types::VectorOfRect,
    videoio::{VideoCapture, CAP_ANY},
    Result,
};

fn main() -> Result<()> {
    // Load Haar cascades for face and eye detection
    let face_cascade = CascadeClassifier::new("haarcascade_frontalface_alt.xml")?;
    let eye_cascade = CascadeClassifier::new("haarcascade_eye.xml")?;

    // Load SVM classifier for pedestrian detection
    let svm_detector = CSVMDetector::new("HOGpedestrians.xml")?;

    // Load a video capture device
    let mut cap = VideoCapture::new(CAP_ANY)?;
    cap.set(opencv::videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
    cap.set(opencv::videoio::CAP_PROP_FRAME_HEIGHT, 480.0)?;

    // Process frames from the video capture device
    loop {
        let mut frame = Mat::default()?;
        cap.read(&mut frame)?;

        // Convert the frame to grayscale
        let mut gray = Mat::default()?;
        cvt_color(&frame, &mut gray, COLOR_BGR2GRAY, 0)?;

        // Detect faces in the grayscale image
        let mut faces = VectorOfRect::new();
        face_cascade.detect_multi_scale(&gray, &mut faces, 1.1, 2, 0, (30, 30))?;

        // Draw rectangles around the detected faces and eyes
        for face in faces {
            rectangle(&mut frame, face, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, 8, 0)?;
            let roi_gray = gray.get_roi(face)?;
            let mut eyes = VectorOfRect::new();
            eye_cascade.detect_multi_scale(&roi_gray, &mut eyes, 1.1, 2, 0, (30, 30))?;
            for eye in eyes {
                let eye_center = face.tl() + roi_gray.tl() + eye.tl() + (eye.width / 2, eye.height / 2);
                let radius = (eye.width + eye.height) as f64 / 4.0;
                opencv::imgproc::circle(
                    &mut frame,
                    eye_center,
                    radius as i32,
                    Scalar::new(255.0, 0.0, 0.0, 0.0),
                    2,
                    8,
                    0,
                )?;
            }
        }

        // Detect pedestrians in the grayscale image
        let mut detections = VectorOfRect::new();
        svm_detector.detect_multi_scale(&gray, &mut detections, 0.0, (8, 8), (32, 32), 1.05, 2.0)?;

        // Draw rectangles around the detected pedestrians
        for detection in detections {
            rectangle(&mut frame, detection, Scalar::new(0.0, 0.0, 255.0, 0.0), 2, 8, 0)?;
        }

        // Show the processed frame
        opencv::highgui::imshow("Video", &mut frame)?;
        opencv::highgui::wait_key(10)?;
    }
}
