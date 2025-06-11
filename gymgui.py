# Required Libraries
import cv2
import time
import numpy as np
import mediapipe as mp
import pygame
from gtts import gTTS
import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window

# Initialize pygame mixer
pygame.mixer.init()
speech_counter = 0
def speak(text):
    global speech_counter
    filename = f"temp_speech_{speech_counter}.mp3"
    tts = gTTS(text=text)
    tts.save(filename)
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    finally:
        pygame.mixer.music.unload()
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                time.sleep(1)
                os.remove(filename)
    speech_counter += 1

# Calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# Base class for counters
class BaseCounter:
    def __init__(self):
        self.reps = 0
        self.dir = 0
    def update(self, lm): return self.reps, False

class SquatCounter(BaseCounter):
    def update(self, lm):
        rep_added = False
        hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        angle = calculate_angle(hip, knee, ankle)
        if angle < 110 and self.dir == 0:
            self.dir = 1
        elif angle > 160 and self.dir == 1:
            self.reps += 1
            self.dir = 0
            rep_added = True
        return self.reps, rep_added

class LegExtensionCounter(BaseCounter):
    def update(self, lm):
        rep_added = False
        hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        angle = calculate_angle(hip, knee, ankle)
        if angle > 160 and self.dir == 0:
            self.dir = 1
        elif angle < 90 and self.dir == 1:
            self.reps += 1
            self.dir = 0
            rep_added = True
        return self.reps, rep_added

class LegCurlCounter(LegExtensionCounter): pass
class LegPressCounter(LegExtensionCounter): pass
class LungeCounter(SquatCounter): pass

class BicepsCurlCounter(BaseCounter):
    def update(self, lm):
        rep_added = False
        shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = calculate_angle(shoulder, elbow, wrist)
        if angle < 40 and self.dir == 0:
            self.dir = 1
        elif angle > 150 and self.dir == 1:
            self.reps += 1
            self.dir = 0
            rep_added = True
        return self.reps, rep_added

class TricepsExtensionCounter(BaseCounter):
    def update(self, lm):
        rep_added = False
        shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = calculate_angle(shoulder, elbow, wrist)
        if angle > 160 and self.dir == 0:
            self.dir = 1
        elif angle < 70 and self.dir == 1:
            self.reps += 1
            self.dir = 0
            rep_added = True
        return self.reps, rep_added

# Select counter based on name
def get_exercise_counter(name):
    name = name.lower()
    if "squat" in name: return SquatCounter()
    if "extension" in name: return LegExtensionCounter()
    if "curl" in name and "leg" in name: return LegCurlCounter()
    if "press" in name: return LegPressCounter()
    if "lunge" in name: return LungeCounter()
    if "biceps" in name or "barbell" in name or "hammer" in name or "preacher" in name: return BicepsCurlCounter()
    if "triceps" in name or "dip" in name: return TricepsExtensionCounter()
    return BaseCounter()

# Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# GUI Layout
class GymLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.exercise_input = TextInput(hint_text='Exercise Name', multiline=False)
        self.sets_input = TextInput(hint_text='Sets', multiline=False, input_filter='int')
        self.reps_input = TextInput(hint_text='Repetitions', multiline=False, input_filter='int')
        self.start_button = Button(text='Start', on_press=self.start_workout)
        self.add_widget(Label(text='AI Gym Support', font_size='24sp'))
        self.add_widget(self.exercise_input)
        self.add_widget(self.sets_input)
        self.add_widget(self.reps_input)
        self.add_widget(self.start_button)

    def start_workout(self, instance):
        exercise = self.exercise_input.text
        sets = int(self.sets_input.text)
        reps = int(self.reps_input.text)
        speak(f"Starting {exercise} for {sets} sets")
        workout_session(exercise, sets, reps)

# Workout Logic
def workout_session(exercise_name, sets, reps):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not found")
        return

    counter = get_exercise_counter(exercise_name)

    for set_num in range(sets):
        speak(f"Start set {set_num + 1}")
        rep_count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                rep_count, added = counter.update(results.pose_landmarks.landmark)
                if added: speak(str(rep_count))
                if rep_count >= reps:
                    speak("Set complete")
                    break
            cv2.imshow("Workout", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if set_num < sets - 1:
            speak("Rest for 2 minutes")
            time.sleep(120)
            reps = max(reps - 2, 1)
    speak("Workout finished, take 3 minutes rest")
    time.sleep(180)
    cap.release()
    cv2.destroyAllWindows()

# Run App
class GymApp(App):
    def build(self):
        Window.size = (480, 800)
        return GymLayout()

if __name__ == '__main__':
    GymApp().run()
