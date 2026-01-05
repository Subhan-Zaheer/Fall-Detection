import os
import cv2
import glob
import traceback

import albumentations as A


from tqdm.notebook import tqdm


def augmentation_naming_convention(frame, aug_count):
    if frame.endswith((".jpg", ".jpeg", ".png")):
        frame_name = frame.split('.')[0]
    
    else:
        frame_name = frame
        
    final_frame_name = frame_name + '_aug_' + str(aug_count)
    
    return final_frame_name
    

def augment_video_frames(input_folder, output_folder, augment, num_aug=3):
    """
    input_folder: original video frames (e.g., person1_fall)
    output_folder: where augmented versions will be stored
    num_aug: how many augmented versions to create
    """
    frames = sorted(glob.glob(f"{input_folder}/*.*g"))

    for frame_name in tqdm(frames):
        frame_path = os.path.join(input_folder, frame_name)
        image = cv2.imread(frame_path)
        if image is None:
            raise FileExistsError(f"File path doesn't exists : {frame_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        for n in range(1, num_aug+1):  # multiple augmented versions
            # Apply augmentation
            augmented = augment(image=image)
            aug_img = augmented["image"]
            temp_frame_name, ext = os.path.splitext(frame_name)
            updated_frame_name = augmentation_naming_convention(temp_frame_name, n)
            updated_frame_name += ext

            # Save
            save_path = os.path.join(output_folder, updated_frame_name)
            if os.path.exists(save_path):
                return
            cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            
            
def run_augmentation(ROOT_DIR):
    height = width =224
    # Define heavy augmentations (to save offline)
    augment = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.GaussianBlur(blur_limit=5, p=0.3),
        A.RandomGamma(p=0.5),
        A.CLAHE(p=0.3),  # adaptive histogram equalization
        A.Perspective(scale=(0.05,0.1), p=0.3),
        A.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), p=0.5)
    ])
    for person_folder in tqdm(os.listdir(ROOT_DIR)):
        person_folder_path = os.path.join(ROOT_DIR, person_folder)
        
        for activity in os.listdir(person_folder_path):
            activity_folder_path = os.path.join(person_folder_path, activity)
            try:
                
                augment_video_frames(activity_folder_path, activity_folder_path, augment)
                
            except Exception as e:
                print(traceback.format_exc())
                print(e)

