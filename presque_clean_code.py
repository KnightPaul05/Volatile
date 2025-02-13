import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import screeninfo



def distance(point1, point2):
    '''retourne la distance entre 2 points'''
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def sdfunc (a,x):
    '''retourne le polynôme évalué en un point'''
    return a[0]*(x**2)+a[1]*x+a[2]

def draw(current_frame, phase):
    '''met à jour les différents éléments sur le fenêtre selon la phase'''
    if phase >= 0:
        cv2.line(current_frame, (int(info[1][0]), int(info[0])), (int(info[2][0]), int(info[0])), (255, 255, 0), 2, cv2.LINE_4)
    if phase >= 1:
        cv2.line(current_frame, (int(info[1][0]), int(info[0])), (int(info[1][0]), int(info[1][1])), (0, 0, 255), 2, cv2.LINE_4)  
    if phase >= 2:
        cv2.line(current_frame, (int(info[2][0]), int(info[0])), (int(info[2][0]), int(info[1][1])), (0, 0, 255), 2, cv2.LINE_4) 
    if phase >= 3:
        cv2.rectangle(current_frame, (int(info[3][0][0]), int(info[3][0][1])), (int(info[3][1][0]), int(info[3][1][1])), (0, 255, 0), 2)
    if phase >= 4:
        cv2.rectangle(current_frame, (int(info[4][0][0]), int(info[4][0][1])), (int(info[4][1][0]), int(info[4][1][1])), (0, 255, 0), 2)

def mouse_event(event, x, y, flags, param):
    '''met à jour les paramètres grace au mouvement de la souris'''
    global phase, dragging, start_x, start_y

    if event == cv2.EVENT_LBUTTONDOWN and not dragging:
        dragging = True
        start_x, start_y = x, y
    if event == cv2.EVENT_LBUTTONUP:
        dragging = False
    if dragging:
        if phase == 0:
            info[0] = y
        if phase == 1:
            info[1] = (x, y)
        if phase == 2:
            info[2] = (x, y)
        if phase == 3:
            info[3][0] = [start_x, start_y]
            info[3][1] = [x, y]
        if phase == 4:
            info[4][0] = [start_x, start_y]
            info[4][1] = [x, y]

def texte(current_frame, phase):
    '''affiche le texte sur la fenêtre selon la phase'''
    cv2.putText(current_frame, titre[phase], (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

def next_phase():
    '''change de phase et attend'''
    global phase
    phase += 1
    cv2.waitKey(100)

phase = 0
dragging = False
start_x, start_y = 0, 0
info = [602, (388, 512), (1408, 518), [[422, 506], [542, 580]], [[1276, 520], [1390, 588]]]
titre = ['Selectionner le haut du fil',
         'Selectionner le haut de la mire gauche',
         'Selectionner le haut de la mire droite',
         'Selectionner la fenetre attaque gauche',
         'Selectionner la fenetre attaque droite']

def set_parametre(frame):
    '''ouvre une fenêtre qui permet de choisir les paramètres de la vidéo avec la souris'''
    global phase
    phase = 0

    while True:
        frame_copy = frame.copy()
        cv2.setMouseCallback('main', mouse_event)
        draw(frame_copy, phase)
        texte(frame_copy, phase)
        cv2.imshow('main', frame_copy) 

        key = cv2.waitKey(1)
        if key == 13:
            next_phase()
        if key == ord('q') or phase == 5:
            break
    
def frame_difference(prev, next):
    '''retourne la différence de frame'''
    if prev is None or next is None:
        return next
    frame_diff = cv2.absdiff(prev, next)
    gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)    
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    #cv2.imwrite('a.png', thresh)
    return thresh

def ouverture(frame):
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    frame = cv2.dilate(frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    #frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    #frame = cv2.dilate(frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    return frame

def mask_on_frame(frame, mask):
    '''retourne l'écran avec le masque appliqué'''
    return cv2.bitwise_and(frame, frame, mask=mask)

def blur(frame, streght):
    '''renvoie une image floutée'''
    return cv2.GaussianBlur(frame, (streght, streght), 0)

def get_contours(frame, para1, para2):
    '''renvoie les contours d'une photo avec Canny'''
    return cv2.Canny(frame, para1, para2)

def circles_Hough(frame, dp, minDist, param1, param2, minRadius, maxRadius):
    '''retourne une liste de cercles apparent à l'image'''
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.HoughCircles(gray_image, 
                            cv2.HOUGH_GRADIENT, 
                            dp=dp, 
                            minDist=minDist, 
                            param1=param1, 
                            param2=param2, 
                            minRadius=minRadius, 
                            maxRadius=maxRadius)

def draw_circle(frame, circles):
    '''dessine les cercles sur l'image'''
    if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

def find_circles(frame, contours, minRound, maxRound, minRadius, maxRadius, ball_found):
    '''ajoute les cercles trouvé sur l'image aux listes'''
    l = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        #cv2.circle(frame, center, radius, (0, 255, 0), 2)

        if minRadius <= radius <= maxRadius and (y < info[0] - (info[0] - info[1][1])/2 and info[1][0] < x < info[2][0]):
            #cv2.circle(frame, center, radius, (0, 255, 0), 2)
            l.append([x, y])
    return l 
                
def mini_liste(ref, liste, limite):
    mini_pt = [0, 0]
    mini = 10000
    for point in liste:
        if point[1] < mini:
            mini = point[1]
            mini_pt = point

    if distance(mini_pt, ref) < limite:
        return mini_pt
    return [0, 0]

def is_ball(l):
    for p1 in l:
        for p2 in l:
            for p3 in l:
                if p1 != p2 and p2 != p3 and p1!= p3 and distance(p1, p2) + distance(p2, p3) < 200 :
                    return [p1, p2, p3]
    return False

def save(name, pic):
    cv2.imwrite(name, pic)

def is_left(l):
    '''renvoie vraie si la balle va vers la gauche, faux sinon'''
    if l[-2] < l[-1]:
        return False
    return True

def actu_stat(stat, high, preci):
    print(high, preci)
    if preci and high:
        stat[0] += 1
    elif not preci and high:
        stat[1] += 1
    elif preci and not high:
        stat[2] += 1
    else :
        stat[3] += 1

def frame_difference2(prev, next):
    # Vérification si les frames sont valides
    if prev is None or next is None:
        raise ValueError("L'une des images est invalide (None). Vérifiez la capture vidéo.")

    # Assurez-vous que les dimensions des images correspondent
    if prev.shape != next.shape:
        next = cv2.resize(next, (prev.shape[1], prev.shape[0]))

    # Assurez-vous que les deux images sont en niveaux de gris
    if len(prev.shape) == 3:
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if len(next.shape) == 3:
        next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    # Calcul de la différence absolue
    frame_diff = cv2.absdiff(prev, next)

    return frame_diff

def debut(chemin_video=None,video_camera=1):
    '''affiche la vidéo avec plein d'infos sympas'''
    if chemin_video is None:
        root = Tk()
        root.withdraw() 
        video_file = askopenfilename(
        title="Sélectionner une vidéo",
        filetypes=[("Fichiers vidéo", "*.mp4 *.avi *.mov *.mkv")])
    else:
        video_file = chemin_video
    if video_camera == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_file)
    

    # Obtenir la taille de l'écran
    screen = screeninfo.get_monitors()[0]  # Premier écran (principal)
    screen_width, screen_height = screen.width, screen.height
                    

    ret, prev_frame = cap.read()
    '''for i in range(1000):
        cap.read()
    ret, prev_frame = cap.read()'''


    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('main', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    prev_frame = cv2.resize(prev_frame, (screen_width, screen_height))

    #set_parametre(prev_frame)

    points = []
    hau_mire = 0
    is_precise = False
    no_ball_frame = 0
    ball_found = False
    is_high = False
    is_precise = False
    start_time = time.time()
    start_traj = 0
    is_start = is_begin = is_fall = False
    stat = [0, 0, 0, 0]
    a = b = time.time()

    while cap.isOpened():        
        time_boucle = time.time() - start_time
        print(time_boucle)
        start_time = time.time()
        #fps = time_boucle/(1/30)
        #start_time = time.time()
        '''for i in range(int(fps)+1):'''
        ret, next_frame = cap.read()
        
        next_frame =cv2.resize(next_frame, (screen_width, screen_height))
        #ret, next_frame = cap.read()

        time_boucle = time.time() - start_time    
        '''if 2/30 - time_boucle > 0:    
            time.sleep(2/30 - time_boucle)'''
        


        #next_frame = cv2.resize(next_frame, (standard_width, standard_height))

        if not ret:
            break


        frame_diff = frame_difference(prev_frame, next_frame)
        frame_diff = ouverture(frame_diff)
       
        
        contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      
               
        all_ball = find_circles(prev_frame, contours, 0.3, 1, 15, 50, ball_found)
        
        if not ball_found:
            points += all_ball
            if is_ball(points) != False:
                points = is_ball(points)
                ball_found = True
                left = is_left(points)
        else:
            next_point = mini_liste(points[-1], all_ball, 100)

            if next_point == [0, 0] or next_point[1] > info[0]:
                no_ball_frame += 1
                #points.append([2*points[-1][0] - points[-2][0], 2*points[-1][1] - points[-2][1]])      
            else:
                no_ball_frame = 0
                points.append(next_point)
                

            if points[-3][1] < points[-2][1] and points[-2][1] < points[-1][1] and not is_begin and not is_start:
                is_begin = True
             
            if points[-3][1] > points[-2][1] and points[-2][1] > points[-1][1] and points[-3][1] > info[0] - (info[0] - info[1][1])*2 and not is_start and is_begin :
                is_start = True
                start_traj = len(points) - 2
                left = is_left(points)
                is_begin = False
            
            if points[-3][1] < points[-2][1] and points[-2][1] < points[-1][1] and is_start and not is_fall:
                is_fall = True
                actu_stat(stat, is_high, is_precise)
            
            if points[-3][1] > points[-2][1] and points[-2][1] > points[-1][1] and is_start and is_fall:
                no_ball_frame = 6
            
            if (points[-3][0] > points[-2][0] and points[-2][0] > points[-1][0] and not left) or (points[-3][0] < points[-2][0] and points[-2][0] < points[-1][0] and left):
                no_ball_frame = 6

 
            z = np.polyfit([points[i][0] for i in range(start_traj, len(points))], [points[i][1] for i in range(start_traj, len(points))], 2)
            
            
            '''if (left and sdfunc(z, info[1][0] + (info[2][0]-info[1][0])/5) < info[0]) or (not left and sdfunc(z, info[2][0] - (info[2][0]-info[1][0])/5) < info[0]) :
                set = True
            else:
                set = False'''

            x_box_1 = np.linspace(info[3][0][0], info[3][1][0], 100)
            x_box_2 = np.linspace(info[4][0][0], info[4][1][0], 100)
            x_box = np.concatenate((x_box_1, x_box_2))
            y_box = sdfunc(z, x_box)

            is_precise = False
            for y in y_box:
                if info[3][0][1] <= y <= info[3][1][1] or info[4][0][1] <= y <= info[4][1][1]:
                    is_precise = True
                    break
            

            hau_mire = (info[0] - (z[2] - (z[1]**2)/(4*z[0])))/(info[0]-info[1][1])
    

            if 2.5 < hau_mire < 3.5:
                is_high = True
            else : is_high = False

            if is_precise and is_high:
                color = (0, 255, 0)
            elif not is_precise and is_high:
                color = (255, 0, 0)
            elif is_precise and not is_high:
                color = (255, 255, 255)
            else :
                color = (0, 0, 255)

            if is_start:
                for i in range(start_traj, len(points)-1):
                    cv2.line(prev_frame, (int(points[i][0]), int(sdfunc(z, points[i][0]))), (int(points[i+1][0]), int(sdfunc(z, points[i+1][0]))), color, 2, cv2.LINE_4)
                    
            if is_start and start_traj < len(points):
                i = 0
                while is_start and not left and sdfunc(z, points[start_traj][0] - (i+1)) < info[0] - 20 and z[0] > 0:
                    cv2.line(prev_frame, (int(points[start_traj][0]-i), int(sdfunc(z, points[start_traj][0]-i))), (int(points[start_traj][0]-(i+1)), int(sdfunc(z, points[start_traj][0]-(i+1)))), color, 2, cv2.LINE_4)
                    i += 1

                i = 0
                while is_start and left and sdfunc(z, points[start_traj][0] + (i+1)) < info[0] - 20 and z[0] > 0:
                    cv2.line(prev_frame, (int(points[start_traj][0]+i), int(sdfunc(z, points[start_traj][0]+i))), (int(points[start_traj][0]+(i+1)), int(sdfunc(z, points[start_traj-1][0]+(i+1)))), color, 2, cv2.LINE_4)
                    i += 1

        white_pixels = cv2.countNonZero(frame_diff)
        total_pixels = frame_diff.size
        white_percentage = (white_pixels / total_pixels) * 100
        if white_percentage > 70:
            no_ball_frame = 6
            is_begin = False

                
        if no_ball_frame >= 6:
            no_ball_frame = 0
            ball_found = False
            is_high = False
            is_precise = False
            start_traj = 0
            is_start = False
            is_fall = False
            points = []
          
        cv2.line(prev_frame, (int(info[1][0]), int(info[0] - 3*(info[0] - info[1][1]))), (int(info[2][0]), int(info[0] - 3*(info[0] - info[1][1]))), (255, 255, 255), 2)
        '''cv2.putText(prev_frame, "Hauteur de mire : " + str(round(hau_mire, 1)),(0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(prev_frame, "Precision : " + str(is_precise), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)'''
        cv2.putText(prev_frame, "Parfait : " + str(stat[0]), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(prev_frame, "Mid : " + str(stat[1]), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(prev_frame, "Mid : " + str(stat[2]), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(prev_frame, "Nul : " + str(stat[3]), (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(prev_frame, str(abs(b-a)), (0, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(prev_frame, info[3][0], info[3][1], (0, 255, 0), 2)
        cv2.rectangle(prev_frame, info[4][0], info[4][1], (0, 255, 0), 2)

        cv2.imshow('main', prev_frame)

        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            break
        if key == ord('s'):
            set_parametre(prev_frame)
        if key == ord('p'):
            while True:
                if cv2.waitKey(1) == ord('p'):
                    break
        if key == ord('d'):
            cap = cv2.VideoCapture(0)
            ret, prev_frame = cap.read()

            points = []
            hau_mire = 0
            is_precise = False
            no_ball_frame = 0
            ball_found = False
            is_high = False
            is_precise = False
            start_time = time.time()
            start_traj = 0
            is_start = is_begin = is_fall = False
            stat = [0, 0, 0, 0]
            a = b = time.time()
        if key == ord('f'):
            root = Tk()
            root.withdraw()  
            video_file = askopenfilename(
            title="Sélectionner une vidéo",
            filetypes=[("Fichiers vidéo", "*.mp4 *.avi *.mov *.mkv")]
                        )
            cap = cv2.VideoCapture(video_file)
            ret, prev_frame = cap.read()

            points = []
            hau_mire = 0
            is_precise = False
            no_ball_frame = 0
            ball_found = False
            is_high = False
            is_precise = False
            start_time = time.time()
            start_traj = 0
            is_start = is_begin = is_fall = False
            stat = [0, 0, 0, 0]
            a = b = time.time()



        # Update previous frame
        prev_frame = next_frame


#debut()