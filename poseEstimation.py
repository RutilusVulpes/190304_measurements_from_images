from scipy.optimize import least_squares as ls
import numpy as np
import sys

class Camera(object):
    def __init__(self,p,f,c):
        self.p = p                   # Pose
        self.f = f                   # Focal Length in Pixels
        self.c = c

    def projective_transform(self,X):

        x = X[:,0]/X[:,2]
        y = X[:,1]/X[:,2]
        u = self.f * x + self.c[0]/2
        v = self.f * y + self.c[1]/2

        u = np.hstack(u)
        v = np.hstack(v)
        return u,v


    def rotational_transform(self, X, p):

        cosAz= np.cos(p[3])
        sinAz= np.sin(p[3])

        cosPch= np.cos(p[4])
        sinPch= np.sin(p[4])

        cosRoll= np.cos(p[5])
        sinRoll= np.sin(p[5])


        T = np.mat([
        [1, 0, 0,-p[0]],
        [0, 1, 0,-p[1]],
        [0, 0, 1,-p[2]],
        [0, 0, 0, 1]])

        Ryaw = np.mat([
        [cosAz, -sinAz, 0, 0],
        [sinAz, cosAz, 0, 0],
        [0, 0, 1, 0]])

        Rpitch = np.mat([
        [1, 0, 0],
        [0, cosPch, sinPch],
        [0, -sinPch, cosPch]])

        Rroll = np.mat([
        [cosRoll, 0, -sinRoll],
        [0, 1, 0 ],
        [sinRoll, 0 , cosRoll]])

        Raxis = np.mat([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1 , 0]])

        C = Raxis @ Rroll @ Rpitch @ Ryaw @ T

        X = X.dot(C.T)

        u,v = self.projective_transform(X)

        return u,v

    def estimate_pose(self,X_gcp,u_gcp,p):
        p_opt = ls(self.residual, p, method='lm',args=(X_gcp,u_gcp))['x']

        return p_opt

    def residual(self,p,X,u_gcp):
        u,v = self.rotational_transform(X,p)
        u = np.squeeze(np.asarray(u - u_gcp[:,0]))
        v = np.squeeze(np.asarray(v - u_gcp[:,1]))
        resid = np.stack((u, v), axis=-1)
        resid = resid.flatten()
        return resid
def main(argv):
    if(len(argv) < 1):
        print("Please submit as arguments GCPs 1")
        return
    gcps  = open(argv[0], "r")
    obs = []
    true = []

    for line in gcps:
        line = line.strip("\n")
        line = line.split(",")
        if(len(line) > 1):
            true.append([float(line[0]),float(line[1])])
            obs.append([float(line[2]),float(line[3]),float(line[4])])

    true = np.asarray(true)
    obs = np.asarray(obs)
    print(true)
    print(obs)
    FOCAL_LENGTH = 2448
    SENSOR_X = 3264
    SENSOR_Y = 2448

    f = FOCAL_LENGTH
    c = np.array([SENSOR_X,SENSOR_Y])
    p_0 = np.array([272558,5193938,1015,10,10,10])

    cam = Camera(p_0,f,c)

    p_opt = cam.estimate_pose(obs,true,p_0)

    print(p_opt)

    for GCP in obs:
        print(cam.rotational_transform(GCP,p_opt))


if __name__=='__main__':
  main(sys.argv[1:])
