import torch
import torch.nn.functional as F
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, input_dims=[8], env=None, n_actions=2, 
            max_size=1_000_000,
            tau=0.005, gamma=0.99,  alpha=2,
            lr_actor=0.0003, lr_critic_value=0.0003, 
            fc1_dims=256, fc2_dims=256, batch_size=256,
            two_critics=True, remove_stochasticity=False):
        self.gamma = gamma
        self.tau = tau
        self.scale = alpha
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.two_critics = two_critics
        self.remove_stochasticity = remove_stochasticity

        self.actor = ActorNetwork(lr_actor, input_dims, n_actions=n_actions,
                    fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='actor', 
                    max_action=env.action_space.high)
                    
        self.value = ValueNetwork(lr_critic_value, input_dims,
                    fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='value')
        self.target_value = ValueNetwork(lr_critic_value, input_dims,
                    fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='target_value')

        self.critic_1 = CriticNetwork(lr_critic_value, input_dims, n_actions=n_actions,
                    fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='critic_1')
        if self.two_critics:
            self.critic_2 = CriticNetwork(lr_critic_value, input_dims, n_actions=n_actions,
                        fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='critic_2')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, train=True):

        if train:
            state = torch.Tensor([observation]).to(self.actor.device)
            actions, _ = self.actor.sample_normal(state, reparameterize=True,
                                                  remove_stochasticity=self.remove_stochasticity)
        else:
            with torch.no_grad():
                state = torch.Tensor([observation]).to(self.actor.device)
                actions, _ = self.actor.sample_normal(state, reparameterize=False, 
                                                      remove_stochasticity=self.remove_stochasticity)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False,
                                                      remove_stochasticity=self.remove_stochasticity)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1.forward(state, actions)
        if self.two_critics:
            q2_new_policy = self.critic_2.forward(state, actions)
            critic_value = torch.min(q1_new_policy, q2_new_policy)
        else:
            critic_value = q1_new_policy
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True,
                                                      remove_stochasticity=self.remove_stochasticity)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)

        if self.two_critics:
            q2_new_policy = self.critic_2.forward(state, actions)
            critic_value = torch.min(q1_new_policy, q2_new_policy)
        else:
            critic_value = q1_new_policy

        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        if self.two_critics:
            self.critic_2.optimizer.zero_grad()

        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        critic_1_loss = F.mse_loss(q1_old_policy, q_hat)
        if self.two_critics:
            q2_old_policy = self.critic_2.forward(state, action).view(-1)
            critic_2_loss = F.mse_loss(q2_old_policy, q_hat)
            critic_loss = 0.5*(critic_1_loss + critic_2_loss)
        else:
            critic_loss = critic_1_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        if self.two_critics:
            self.critic_2.optimizer.step()

        self.update_network_parameters()
