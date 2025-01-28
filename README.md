# Construction Site Safety Hardhat Detection

## Project Overview

Ensuring the safety of workers on construction sites is critical, and the proper use of Personal Protective Equipment (PPE) like hardhats plays a vital role in preventing injuries and fatalities. However, manually monitoring PPE compliance is time-consuming and susceptible to human error. This project addresses this challenge by developing an automated system that detects whether workers are wearing appropriate PPE, specifically hardhats, using advanced computer vision techniques.

Leveraging the **YOLOv8** model, the system achieves real-time detection with high accuracy and efficiency, making it suitable for dynamic and crowded construction environments. The project also evaluates alternative models such as **Faster R-CNN** and custom **CNNs**, demonstrating the superior performance of YOLOv8 in balancing speed and precision. Trained on the comprehensive Construction Site Safety Image Dataset from Roboflow, the model effectively identifies hardhats, non-compliance instances, and contextual elements like persons and machinery.

## Key Features

- **Real-Time Detection:** Utilizes YOLOv8 for fast and accurate hardhat detection, ensuring immediate identification of PPE compliance.
- **Automated Monitoring:** Reduces the need for manual supervision by continuously tracking PPE usage on construction sites.
- **Robust Performance:** Handles various environmental conditions and worker orientations, maintaining high precision and recall.
- **Scalable Deployment:** Designed for easy integration into existing safety protocols and scalable across multiple sites.

## Technologies Used

- **YOLOv8:** Advanced object detection model for real-time performance.
- **Faster R-CNN:** Evaluated as a comparative model for object detection tasks.
- **PyTorch:** Deep learning framework used for model training and evaluation.
- **OpenCV:** Utilized for video processing and visualization.
- **Roboflow:** Source of the comprehensive Construction Site Safety Image Dataset.

## Getting Started

To learn more about the project setup, installation, and usage, please refer to the [Documentation](./docs).

## Contributions

Contributions are welcome! Please see the [Contributing](./CONTRIBUTING.md) guidelines for more details.

## License

This project is licensed under the [MIT License](./LICENSE).

