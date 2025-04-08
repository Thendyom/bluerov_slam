import numpy as np

class EKF:
    def __init__(self, dt=0.1):
        self.dt = dt

        #extended state vector with orientation:
        #[x, y, z, vx, vy, vz, ax_bias, ay_bias, az_bias, qx, qy, qz, qw]
        self.x = np.zeros((13, 1))
        self.x[12] = 1.0  #initialize as unit quaternion

        #state covariance
        self.P = np.eye(13) * 0.1

        #transition matrix
        self.F = np.eye(13)
        for i in range(3):
            self.F[i, i+3] = dt  #relate position to velocity
            self.F[i+3, i+6] = -dt  #effect of bias on velocity

        #control input model (acceleration)
        self.B = np.zeros((13, 3))
        for i in range(3):
            self.B[i, i] = 0.5 * dt**2  #integrating acceleration for position
            self.B[i+3, i] = dt  #relating acceleration to velocity

        #process noise
        self.Q = np.eye(13) * 0.1

        #measurement matrix for dvl
        self.H_dvl = np.zeros((3, 13))
        self.H_dvl[0, 3] = 1  #velocity in x
        self.H_dvl[1, 4] = 1  #velocity in y
        self.H_dvl[2, 5] = 1  #velocity in z
        self.R_dvl = np.eye(3) * 0.05  #measure noise for dvl

        #measurement matrix for depth
        self.H_depth = np.zeros((1, 13))
        self.H_depth[0, 2] = 1  #depth measurement
        self.R_depth = np.array([[0.1]])  #measure noise for depth

    #quaternion normalization for unit norm
    def normalize_quaternion(self):
        q = self.x[9:13]
        norm = np.linalg.norm(q)
        if norm > 0:
            self.x[9:13] = q / norm  #normalize quaternion

    #predict orientation using gyroscope data
    def predict_orientation(self, gyro, dt):
        wx, wy, wz = gyro
        #for alexandru: skew-symmetric matrix (omega matrix)
        #using this matrix to compute quaternion derivative
        omega = 0.5 * np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        q = self.x[9:13].reshape(4, 1)
        dq = omega @ q * dt  #compute quaternion derivative
        self.x[9:13] = (q + dq)  #update quaternion
        self.normalize_quaternion()  #quaternion normalized

    #predict state based on IMU and optional gyro data
    def predict(self, imu_accel, gyro=None, dt=None):
        if dt is not None and dt != self.dt:
            self.dt = dt  #update time step
            #update transition and control matrices
            for i in range(3):
                self.F[i, i+3] = dt
                self.F[i+3, i+6] = -dt
                self.B[i, i] = 0.5 * dt**2
                self.B[i+3, i] = dt

        bias = self.x[6:9]  #extracting bias values
        u = np.reshape(imu_accel, (3, 1)) - bias  #remove bias from accel
        self.x = self.F @ self.x + self.B @ u

        if gyro is not None:
            self.predict_orientation(gyro, self.dt)  #update orientation if gyro is there

        self.P = self.F @ self.P @ self.F.T + self.Q  #update state covariance

    #update state with measurement
    def update(self, z, H, R):
        y = z - H @ self.x  #measurement residual
        S = H @ self.P @ H.T + R  #innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  #Kalman gain
        self.x = self.x + K @ y  #update state estimate
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P #covariance

    #update state with dvl measurements
    def update_dvl(self, dvl_velocity):
        z = np.reshape(dvl_velocity, (3, 1))  #reshape dvl velocity
        self.update(z, self.H_dvl, self.R_dvl)  #update with dvl data

    #update state with depth measurement
    def update_depth(self, depth_z):
        z = np.array([[depth_z]])  #create measurement array for depth
        self.update(z, self.H_depth, self.R_depth)  #update with depth data

    #current state, output as array
    def get_state(self):
        return self.x.flatten()  #1D array
