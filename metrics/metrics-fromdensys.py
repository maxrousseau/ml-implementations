#!/usr/bin/env python
#-*- coding: utf-8 -*-
import sys
import os
import uuid
import json
import cv2
import dlib
import numpy as np
from modules import linear as ln

class Job(object):
    """This class will act as a controller for the analysis pushed to the api"""


    def __init__(self, image_url, task):
        """Job object to create, store, exectute and post results
        Upon initialization a hash will be assigned to the image which will serve
        as a unique identifier.

        It has two methods create (which will store a new job in the data file) and
        execute which calls the appropriate analysis on an image.

        Task list [med, ratio1, ratio2, ratio3, lfh, asym]

        Parameters
        ----------
        image : string
            path to image uploaded by the user
        task :  int
            int corresponding to the specific analysis requested by the user

        Returns
        ------
        None
        """
        # modification to the default encoder for uuid serialization
        json.JSONEncoder_olddefault = json.JSONEncoder.default
        def JSONEncoder_newdefault(self, o):
            if isinstance(o, uuid.UUID): return str(o)
            return json.JSONEncoder_olddefault(self, o)
        json.JSONEncoder.default = JSONEncoder_newdefault

        # create dictionnary of paths used in this file
        self.hash = uuid.uuid4()
        curr_dir = os.path.dirname(os.path.abspath((__file__)))
        error_log = os.path.join(root_path, "log/errors.log")
        prototxt = os.path.join(root_path, "models/deploy2.prototxt")
        model = os.path.join(root_path, "models/deploy2.caffemodel")
        shape_model = os.path.join(root_path, "models/68_landmarks.dat")
        haar_xml = os.path.join(root_path, "models/haarcascade_frontalface_default.xml")
        joblist = os.path.join(root_path, str("db/jobs/"+str(self.hash)+".json"))
        img_db = os.path.join(root_path, "db", "img")
        self.paths = {
            "error_log": error_log,
            "curr_dir" : curr_dir,
            "prototxt" : prototxt,
            "model" : model,
            "haar_xml" : haar_xml,
            "joblist" : joblist,
            "shape_model" : shape_model,
            "img_db" : img_db
        }
        self.image_url = image_url
        self.task = task
        self.is_complete = False
        # initialize the json object for this job
        self.json_obj = {
            "b64_img" : "NONE",
            "hash" : self.hash,
            "img_url" : self.image_url,
            "resultimg_path": "NONE",
            "task" : self.task,
            "is_complete" : self.is_complete,
            "face_coord" : [],
            "ldmk_coord" : [],
            "asym" : 0,
            "lfh" : 0,
            "MED" : 0,
            "ratio1" : 0,
            "ratio2" : 0,
            "ratio3" : 0,
            "result" : 0
        }
        self.confidence = 0.7
        self.purple = (153, 51, 255)
        self.pink = (255, 51, 255)
        self.blue = (255, 153, 51)
        self.error_log = []

    def url_to_image(self, url):
        wbimg = ul.urlopen(url)
        image = np.asarray(bytearray(wbimg.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image

    def draw_line(self, inpt_img, point_1, point_2, color):
        """Draw a colored line
        This method will draw a colored line on top of the input image and
        return the resulting image.

        Parameters
        ----------
        inpt_img : opencv_image
            input image to be drawn
        point_1 : int array
            coordinate of the the first point (ex. x,y [108, 395])
        point_2 : int array
            coordinate of the the second point (ex. x,y [108, 395])
        color : int tuples
            color of the line to be drawn (ex. BGR (250, 0, 0))

        Returns
        ------
        img_newline : opencv_image
            new image with the line drawn
        """
        height, width = inpt_img.shape[:2]
        adj_thickness = int(height/270)
        img_newline = cv2.line(inpt_img, point_1,
                              point_2, color,
                              thickness=adj_thickness, lineType=cv2.LINE_AA)

        return img_newline

#    def med(self):
#        """Executes the Mean Euclidean Distance analysis"""
#
#        return result

    def b64_encoder(self, img):
        success, encoded_img = cv2.imencode('.png', img)
        if success:
            return base64.b64encode(encoded_img).decode()
        return ''

    def ratio1(self, coords):
        """Computes the value of ratio 1
        This method will compute first facial ratio (37-46/1-17)

        Parameters
        ----------
        coords : numpy array
            coordinates of the 68 landmarks

        Returns
        ------
        ratio1_res : int
            result from the computation of the first ratio
        """
        bitemp_dist = ln.Linear.euc_dist(self, coords["X1"], coords["Y1"],
                                      coords["X17"], coords["Y17"])
        bioc_dist = ln.Linear.euc_dist(self, coords["X37"], coords["Y37"],
                                      coords["X46"], coords["Y46"])
        ratio1_res = bioc_dist / bitemp_dist
        self.json_obj["ratio1"] = ratio1_res

        point_1 = (coords["X1"], coords["Y1"])
        point_2 = (coords["X17"], coords["Y17"])
        result_image = image
        result_image = self.draw_line(result_image, point_1, point_2, self.blue)

        point_1 = (coords["X37"], coords["Y37"])
        point_2 = (coords["X46"], coords["Y46"])
        result_image = self.draw_line(result_image, point_1, point_2, self.purple)

        self.json_obj["b64_img"] = self.b64_encoder(result_image)

        img_name = os.path.join(self.paths["img_db"], str(self.hash)+"_ratio1.png")
        cv2.imwrite(img_name, result_image)
        self.json_obj["resultimg_path"] = img_name

        return ratio1_res

    def ratio2(self, coords):
        """Computes the value of ratio 2
        This method will compute second facial ratio (5-13/1-17)

        Parameters
        ----------
        coords : numpy array
            coordinates of the 68 landmarks

        Returns
        ------
        ratio1_res : int
            result from the computation of the first ratio
        """
        bitemp_dist = ln.Linear.euc_dist(self, coords["X1"], coords["Y1"],
                                      coords["X17"], coords["Y17"])
        bimand_dist = ln.Linear.euc_dist(self, coords["X5"], coords["Y5"],
                                      coords["X13"], coords["Y13"])
        ratio2_res = bimand_dist / bitemp_dist
        self.json_obj["ratio2"] = ratio2_res

        point_1 = (coords["X1"], coords["Y1"])
        point_2 = (coords["X17"], coords["Y17"])
        result_image = image
        result_image = self.draw_line(result_image, point_1, point_2, self.blue)

        point_1 = (coords["X5"], coords["Y5"])
        point_2 = (coords["X13"], coords["Y13"])
        result_image = self.draw_line(result_image, point_1, point_2, self.purple)
        self.json_obj["b64_img"] = self.b64_encoder(result_image)

        img_name = os.path.join(self.paths["img_db"], str(self.hash)+"_ratio2.png")
        cv2.imwrite(img_name, result_image)
        self.json_obj["resultimg_path"] = img_name

        return ratio2_res

    def ratio3(self, coords):
        """Computes the value of ratio 3
        This method will compute third facial ratio (1-17/28-9)

        Parameters
        ----------
        coords : numpy array
            coordinates of the 68 landmarks

        Returns
        ------
        ratio3_res : int
            result from the computation of the third ratio
        """
        bitemp_dist = ln.Linear.euc_dist(self, coords["X1"], coords["Y1"],
                                      coords["X17"], coords["Y17"])
        face_height = ln.Linear.euc_dist(self, coords["X28"], coords["Y28"],
                                      coords["X9"], coords["Y9"])
        ratio3_res = bitemp_dist / face_height
        self.json_obj["ratio3"] = ratio3_res

        point_1 = (coords["X1"], coords["Y1"])
        point_2 = (coords["X17"], coords["Y17"])
        result_image = image
        result_image = self.draw_line(result_image, point_1, point_2, self.blue)

        point_1 = (coords["X28"], coords["Y28"])
        point_2 = (coords["X9"], coords["Y9"])
        result_image = self.draw_line(result_image, point_1, point_2, self.purple)
        self.json_obj["b64_img"] = self.b64_encoder(result_image)

        img_name = os.path.join(self.paths["img_db"], str(self.hash)+"_ratio3.png")
        cv2.imwrite(img_name, result_image)
        self.json_obj["resultimg_path"] = img_name

        return ratio3_res

    def lfh(self, coords):
        """Computes the lower face height
        This method will computes the lower face height (34-9/28-9)

        Parameters
        ----------
        coords : numpy array
            coordinates of the 68 landmarks

        Returns
        ------
       lfw_res : int
            result from the computation of the lower face height
        """
        lower_face = ln.Linear.euc_dist(self, coords["X34"], coords["Y34"],
                                      coords["X9"], coords["Y9"])
        face_height = ln.Linear.euc_dist(self, coords["X28"], coords["Y28"],
                                      coords["X9"], coords["Y9"])
        lfw_res = lower_face / face_height
        self.json_obj["lfh"] = lfw_res

        point_1 = (coords["X34"], coords["Y34"])
        point_2 = (coords["X9"], coords["Y9"])
        result_image = image
        result_image = self.draw_line(result_image, point_1, point_2, self.blue)

        point_1 = (coords["X28"], coords["Y28"])
        point_2 = (coords["X9"], coords["Y9"])
        result_image = self.draw_line(result_image, point_1, point_2, self.purple)
        self.json_obj["b64_img"] = self.b64_encoder(result_image)

        img_name = os.path.join(self.paths["img_db"], str(self.hash)+"_lfh.png")
        cv2.imwrite(img_name, result_image)
        self.json_obj["resultimg_path"] = img_name


        return lfw_res


    def asym(self, coords):
        """Compute the asymmetry index
        This method will compute the asymmetry index by comparing corresponding
        landmark distances to the facial midline (nasion - menton).

        Parameters
        ----------
        coords : numpy array
            coordinates of the 60 landmarks

        Returns
        ------
        sum_diff : int
            sum of the differences from the coordinates
        """
        # begin by computing f(x) for the midline
        nasion_x = np.array([coords["X9"]])
        nasion_y = np.array([coords["Y9"]])
        menton_x = np.array([coords["X28"]])
        menton_y = np.array([coords["Y28"]])
        midline = ln.Linear(nasion_x, nasion_y, menton_x, menton_y)

        # compute f(x) R-L for corresponding landmarks
        corrsp_ax = np.array([coords["X1"], coords["X2"], coords["X3"],
                              coords["X4"], coords["X5"], coords["X6"],
                              coords["X7"], coords["X8"], coords["X18"],
                              coords["X19"], coords["X20"], coords["X21"],
                              coords["X22"], coords["X32"], coords["X33"],
                              coords["X37"], coords["X38"], coords["X39"],
                              coords["X40"], coords["X41"], coords["X42"]])

        corrsp_ay = np.array([coords["Y1"], coords["Y2"], coords["Y3"],
                              coords["Y4"], coords["Y5"], coords["Y6"],
                              coords["Y7"], coords["Y8"], coords["Y18"],
                              coords["Y19"], coords["Y20"], coords["Y21"],
                              coords["Y22"], coords["Y32"], coords["Y33"],
                              coords["Y37"], coords["Y38"], coords["Y39"],
                              coords["Y40"], coords["Y41"], coords["Y42"]])

        corrsp_bx = np.array([coords["X17"], coords["X16"], coords["X15"],
                              coords["X14"], coords["X13"], coords["X12"],
                              coords["X11"], coords["X10"], coords["X27"],
                              coords["X26"], coords["X25"], coords["X24"],
                              coords["X23"], coords["X36"], coords["X35"],
                              coords["X46"], coords["X45"], coords["X44"],
                              coords["X43"], coords["X48"], coords["X47"]])

        corrsp_by = np.array([coords["Y17"], coords["Y16"], coords["Y15"],
                              coords["Y14"], coords["Y13"], coords["Y12"],
                              coords["Y11"], coords["Y10"], coords["Y27"],
                              coords["Y26"], coords["Y25"], coords["Y24"],
                              coords["Y23"], coords["Y36"], coords["Y35"],
                              coords["Y46"], coords["Y45"], coords["Y44"],
                              coords["Y43"], coords["Y48"], coords["Y47"]])

        lin_func = ln.Linear(corrsp_ax, corrsp_ay, corrsp_bx, corrsp_by)

        # compute the coordinates of the origins (O)
        origin_x, origin_y = lin_func.solve(midline.slopes, midline.constants)


        # compute distance RO and LO
        distances_a = lin_func.euc_dist(corrsp_ax, corrsp_ay, origin_x, origin_y)
        distances_b = lin_func.euc_dist(corrsp_bx, corrsp_by, origin_x, origin_y)

        # compute absolute difference RO - LO
        diff_ab = (distances_a - distances_b) / (distances_a + distances_b)
        abs_diff = np.absolute(diff_ab)

        # sum of absolute differences
        sum_diff = np.sum(abs_diff)
        self.json_obj["asym"] = sum_diff

        result_image = image

        for i in range(len(corrsp_ax)):
            point_1 = (corrsp_ax[i], corrsp_ay[i])
            point_2 = (int(origin_x[i]), int(origin_y[i]))
            result_image = self.draw_line(result_image, point_1, point_2, self.blue)

            point_1 = (corrsp_bx[i], corrsp_by[i])
            result_image = self.draw_line(result_image, point_1, point_2, self.purple)

        self.json_obj["b64_img"] = self.b64_encoder(result_image)

        img_name = os.path.join(self.paths["img_db"], str(self.hash)+"_asym.png")
        cv2.imwrite(img_name, result_image)
        self.json_obj["resultimg_path"] = img_name

        return sum_diff

    def execute(self):
        """Method creating a new job entry to be saved in joblist.json
        Here we call the appropriate analysis depending on the task that was

        This method will store the job's hash, task, image path and current job
        status once completed.

        Parameters
        ----------
        None

        Returns
        ------
        result : string
            result of the specific analysis presented as a string.
        """

        result_str = None
        result_int = 0
        #self.error_log.append("ran")


        ###########################################################
        # face detetion: load network - prepare image - run network
        ###########################################################

        face_net = cv2.dnn.readNetFromCaffe(self.paths["prototxt"],
                                            self.paths["model"])
        global image
        image = self.url_to_image(self.image_url)
        (height, width) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                     1.0,
                                     (300, 300),
                                     (103.93, 116.77, 123.68))
        face_net.setInput(blob)
        detections = face_net.forward()

        count = 0;
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence:

                # multiple face warning!!!! --> if detections.shape[2] != 0
                if count > 1:
                    self.error_log.append("[ERROR] NO OR MULTIPLE FACES DETECTED IN IMAGE")

                else:
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width,
                                                             height])
                    (startX, startY, endX, endY) = box.astype("int")
                    np_int_array = [startX, startY, endX, endY]
                    int32_array = []
                    for a in np_int_array:
                        int32_array.append(int(a))
                    self.json_obj["face_coord"] = int32_array
                    self.error_log.append("[INFO] CNN RAN SUCCESSFULLY")
                    count += 1

        ###########################################################
        # facial landmarks: load face - create predictor - annotate
        ###########################################################
        dlib_rect = dlib.rectangle(int32_array[0],
                                   int32_array[1],
                                   int32_array[2],
                                   int32_array[3])

        # Haar cascade is used as a safeguard
        if count != 1:
            face_cascade = cv2.CascadeClassifier(self.paths["haar_xml"])
            faces_detect = face_cascade.detectMultiScale(image,
                                                         scaleFactor=1.1,
                                                         minSize=(100,100),
                                                         maxSize=(500,500),
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
            # let user know of multiple faces detected in one image
            if len(faces_detect) > 1:
                self.error_log.append("[ERROR] {} FACES DETECTED IN {}".format(len(faces_detect), self.json_obj["hash"]))

            else:
                pass

            x1, y1, w, h = faces_detect[0]
            x2 = x1 + w
            y2 = y1 + h

            dlib_rect = dlib.rectangle(x1, y1, x2, y2)
            self.error_log.append("[INFO] HAAR CASCADE RAN SUCCESSFULLY")

        predictor = dlib.shape_predictor(self.paths["shape_model"])

        landmarks = predictor(image, dlib_rect).parts()

        tuples_array = [(p.x, p.y) for p in landmarks]
        ldmk_coords = {}
        count = 1
        for x,y in tuples_array:
            x_coord = "X"+str(count)
            y_coord = "Y"+str(count)
            ldmk_coords[x_coord] = x
            ldmk_coords[y_coord] = y
            count += 1

        self.json_obj["ldmks_coords"] = ldmk_coords


        # run analysis functions
        if self.json_obj["task"] == "asym":
            asym_result = self.asym(ldmk_coords)
            self.json_obj["result"] = asym_result
            result_str = str(asym_result)

        elif self.json_obj["task"] == "lfh":
            lfh_result = self.lfh(ldmk_coords)
            self.json_obj["result"] = lfh_result
            result_str = str(lfh_result)

#        elif self.json_obj["task"] == "med":
#           None

        elif self.json_obj["task"] == "ratio1":
            ratio1_result = self.ratio1(ldmk_coords)
            self.json_obj["result"] = ratio1_result
            result_str = str(ratio1_result)

        elif self.json_obj["task"] == "ratio2":
            ratio2_result = self.ratio2(ldmk_coords)
            self.json_obj["result"] = ratio2_result
            result_str = str(ratio2_result)

        elif self.json_obj["task"] == "ratio3":
            ratio3_result = self.ratio3(ldmk_coords)
            self.json_obj["result"] = ratio3_result
            result_str = str(ratio3_result)

        else:
            print("[ERROR] TASK DOES NOT EXIST")
            print("[INFO] cancelling job")

        # write results to an outfile
        if not os.path.isfile(self.paths["joblist"]):
            with open(self.paths["joblist"], "w") as outfile:
                json.dump(self.json_obj, outfile)

        with open(self.paths["error_log"], 'w') as f:
            for err in self.error_log:
                f.write("%s\n" % err)

        return result_str


