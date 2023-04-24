# This code is proposed as a reference solution for various exercises of Home Assignements for the OReL course in 2023.
# This solution is tailored for simplicity of understanding and is in no way optimal, nor the only way to implement the different elements!
import numpy as np



####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

#																	ENVIRONMENTS

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################




# A simple 4-room gridworld implementation with a grid of 7x7 for a total of 20 states (the walls do not count!).
# We arbitrarily chose the actions '0' = 'go up', '1' = 'go right', '2'  = 'go down' thus '3' = 'go left'
# Finally the state '0' is the top-left corner, 'nS - 1' is the down-right corner.
# The agent is teleported back to the the initial state '0' (top-left corner) ,  whenever performing any action in rewarding state '19' (down-right corner).
class four_room():

	def __init__(self):
		self.nS = 20
		nS = self.nS
		self.nA = 4

		self.map = [[-1, -1, -1, -1, -1, -1, -1],
					[-1,  0,  1,  2,  3,  4, -1],
					[-1,  5,  6, -1,  7,  8, -1],
					[-1,  9, -1, -1, 10, -1, -1],
					[-1, 11, 12, 13, 14, 15, -1],
					[-1, 16, 17, -1, 18, 19, -1],
					[-1, -1, -1, -1, -1, -1, -1]]
		map = np.array(self.map)

		# We build the transitions matrix P using the map.
		self.P = np.zeros((nS, 4, nS))
		for s in range(nS):
			temp = np.where(s == map)
			x, y = temp[0][0], temp[1][0]
			up = map[x-1, y]
			right = map[x, y+1]
			down = map[x+1, y]
			left = map[x, y-1]

			# Action 0: go up.
			a = 0
			self.P[s, a, s] += 0.1
			# Up
			if up == -1:
				self.P[s, a, s] += 0.7
			else:
				self.P[s, a, up] += 0.7
			# Right
			if right == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, right] += 0.1
			# Left
			if left == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, left] += 0.1
			
			# Action 1: go right.
			a = 1
			self.P[s, a, s] += 0.1
			# Up
			if up == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, up] += 0.1
			# Right
			if right == -1:
				self.P[s, a, s] += 0.7
			else:
				self.P[s, a, right] += 0.7
			# Down
			if down == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, down] += 0.1
			
			# Action 2: go down.
			a = 2
			self.P[s, a, s] += 0.1
			# Right
			if right == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, right] += 0.1
			# Down
			if down == -1:
				self.P[s, a, s] += 0.7
			else:
				self.P[s, a, down] += 0.7
			# Left
			if left == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, left] += 0.1

			# Action 3: go left.
			a = 3
			self.P[s, a, s] += 0.1
			# Up
			if up == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, up] += 0.1
			# Down
			if down == -1:
				self.P[s, a, s] += 0.1
			else:
				self.P[s, a, down] += 0.1
			# Left
			if left == -1:
				self.P[s, a, s] += 0.7
			else:
				self.P[s, a, left] += 0.7
			
			# Set to teleport back when in the rewarding state.
			if s == self.nS - 1:
				for a in range(4):
					for ss in range(self.nS):
						self.P[s, a, ss] = 0
						if ss == 0:
							self.P[s, a, ss] = 1

			
		# We build the reward matrix R.
		self.R = np.zeros((nS, 4))
		for a in range(4):
			self.R[nS - 1, a] = 1

		# We (arbitrarily) set the initial state in the top-left corner.
		self.s = 0

	# To reset the environment in initial settings.
	def reset(self):
		self.s = 0
		return self.s

	# Perform a step in the environment for a given action. Return a couple state, reward (s_t, r_t).
	def step(self, action):
		new_s = np.random.choice(np.arange(self.nS), p=self.P[self.s, action])
		reward = self.R[self.s, action]
		self.s = new_s
		return new_s, reward








####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

#																	VI and PI

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################






# An implementation of the Value Iteration algorithm for a given environment 'env' and discout 'gamma' < 1.
# An arbitrary 'max_iter' is a maximum number of iteration, usefull to catch any error in your code!
# Return the number of iterations, the final value and the optimal policy.
def VI(env, gamma = 0.9, max_iter = 10**3, epsilon = 10**(-2)):

	# The variable containing the optimal policy estimate at the current iteration.
	policy = np.zeros(env.nS, dtype=int)
	niter = 0

	# Initialise the value and epsilon as proposed in the course.
	V0 = np.array([1/(1 - gamma) for _ in range(env.nS)])
	V1 = np.zeros(env.nS)
	epsilon = epsilon * (1 - gamma) / (2 * gamma)

	# The main loop of the Value Iteration algorithm.
	while True:
		niter += 1
		for s in range(env.nS):
			for a in range(env.nA):
				temp = env.R[s, a] + gamma * sum([V * p for (V, p) in zip(V0, env.P[s, a])])
				if (a == 0) or (temp > V1[s]):
					V1[s] = temp
					policy[s] = a
		
		# Testing the stopping criterion (+1 abitrary stop when 'max_iter' is reached).
		if np.linalg.norm(V1 - V0) < epsilon:
			return niter, V0, policy
		else:
			V0 = V1
			V1 = np.array([1/(1 - gamma) for _ in range(env.nS)])
		if niter > max_iter:
			print("No convergence in VI after: ", max_iter, " steps!")
			return niter, V0, policy




# A first implementation of the PI algorithms, using a matrix inversion to do the policy evaluation step.
def PI(env, gamma = 0.9):

	# Initialisation of the variables.
	policy0 = np.random.randint(env.nA, size = env.nS)
	policy1 = np.zeros(env.nS, dtype = int)
	niter = 0

	# The main loop of the PI algorithm.
	while True:
		niter += 1

		# Policy evaluation step.
		P_pi = np.array([[env.P[s, policy0[s], ss] for ss in range(env.nS)] for s in range(env.nS)])
		R_pi = np.array([env.R[s, policy0[s]] for s in range(env.nS)])
		V0 = np.linalg.inv((np.eye(env.nS) - gamma * P_pi)) @ R_pi
		V1 = np.zeros(env.nS)

		# Updating the policy.
		for s in range(env.nS):
			for a in range(env.nA):
				temp = env.R[s, a] + gamma * sum([u * p for (u, p) in zip(V0, env.P[s, a])])
				if (a == 0) or (temp > V1[s]):
					V1[s] = temp
					policy1[s] = a

		# Testing if the policy changed or not.
		test = True
		for s in range(env.nS):
			if policy0[s] != policy1[s]:
				test = False
				break
		if test:
			return niter, policy1
		else:
			policy0 = policy1
			policy1 = np.zeros(env.nS, dtype=int)




# An auxiliary function for the following algorithm.
# This an algorithmique alternative to the policy evaluation step of the PI.
# Nearly identical to the VI algorithm!
def policy_eval(policy, env, gamma, epsilon, max_iter):

	# Initialisations.
	niter = 0
	V0 = np.array([1/(1 - gamma) for _ in range(env.nS)])
	epsilon = epsilon * (1 - gamma) / (2 * gamma)
	V1 = np.zeros(env.nS)

	# A loop similar to the VI, but with a fixed policy.
	while True:
		niter += 1
		for s in range(env.nS):
			a = policy[s]
			V1[s] = env.R[s, a] + gamma * sum([V * p for (V, p) in zip(V0, env.P[s, a])])

		# The stopping criterion and an arbitrary limit on the number of iterations.
		if np.linalg.norm(V1 - V0) < epsilon:
			return V0
		else:
			V0 = V1
			V1 = np.array([1/(1 - gamma) for _ in range(env.nS)])
		if niter > max_iter:
			print("No convergence in policy evaluation!")
			return V0
	

# Another possible implementation of the PI, using an algorithmique policy evaluation instead of the matrix inversion.
# While it does introduce numerical estimation and thus some incertitude with epsilon too big, it can also prove computationally
# more efficient!
def PI_alternative(env, gamma = 0.9, max_iter = 10**5, epsilon = 10**(-2)):

	# Initialisation of the variables.
	policy0 = np.random.randint(env.nA, size = env.nS)
	policy1 = np.zeros(env.nS, dtype = int)
	niter = 0

	# The main loop of the PI algorithm.
	while True:
		niter += 1

		# Policy evaluation step.
		V0 = policy_eval(policy0, env, gamma, epsilon, max_iter)
		V1 = np.zeros(env.nS)

		# Updating the policy.
		for s in range(env.nS):
			for a in range(env.nA):
				temp = env.R[s, a] + gamma * sum([u * p for (u, p) in zip(V0, env.P[s, a])])
				if (a == 0) or (temp > V1[s]):
					V1[s] = temp
					policy1[s] = a

		# Testing if the policy changed or not.
		test = True
		for s in range(env.nS):
			if policy0[s] != policy1[s]:
				test = False
				break
		if test:
			return niter, policy1
		else:
			policy0 = policy1
			policy1 = np.zeros(env.nS, dtype=int)








####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

#																	Running experiments

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################



env = four_room()
print(VI(env))
