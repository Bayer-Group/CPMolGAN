import math
from scipy import linalg
import logging

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, concatenate, RNN, GRUCell
from tensorflow.keras.layers import  Activation, Dropout, Subtract 
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LeakyReLU

from cpmolgan.utils import *


def build_encoder_decoder(num_tokens, latent_dim= 256, weights=None, verbose=True):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_tokens))
    gru_cells = [GRUCell(hidden_dim) for hidden_dim in [256,256,256]]
    gru_encoder = RNN(gru_cells, return_state=True)
    encoder_outputs_and_states = gru_encoder(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]
    encoder_states = Concatenate(axis=1)(encoder_states)

    z_mean = Dense(latent_dim, name='z_mean', activation = 'linear')(encoder_states)
    z_log_var = Dense(latent_dim, name='z_log_var', activation = 'linear')(encoder_states)

    def sampling(args):
        z_mean_, z_log_var_ = args
        batch_size = K.shape(z_mean_)[0]
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev = 1.)
        return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

    z_mean_log_var_output = Concatenate(name='KL')([z_mean, z_log_var])
    sampled_encoder_states = Lambda(sampling, output_shape=(latent_dim,), name='sampled')([z_mean, z_log_var])
    sampled_encoder_states = Activation('tanh')(sampled_encoder_states)
    z_mean_tanh = Activation('tanh')(z_mean)

    def vae_loss(dummy_true, x_mean_log_var_output):
        x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
        kl_loss = - 0.5 * K.mean(1 + x_log_var - K.square(x_mean) - K.exp(x_log_var), axis = -1)
        return kl_loss

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_tokens))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    Decoder_Dense_Initial = Dense(sum([256,256,256]), activation=None, name = 'Decoder_Dense_Initial')
    decoder_cell_inital = Decoder_Dense_Initial(sampled_encoder_states)
    spliter = Lambda(lambda x: tf.split(x,[256,256,256],axis=-1), name='split')
    decoder_cell_inital = spliter(decoder_cell_inital)

    decoder_cells = [GRUCell(hidden_dim) for hidden_dim in [256,256,256]]
    decoder_gru = RNN(decoder_cells, return_sequences=True, return_state=True)
    decoder_outputs_and_states = decoder_gru(decoder_inputs, initial_state=decoder_cell_inital)
    decoder_outputs = decoder_outputs_and_states[0]
    decoder_outputs = Dropout(0.2)(decoder_outputs)
    Decoder_Dense = Dense(num_tokens, activation='softmax', name = 'Decoder_Dense')
    Decoder_Time_Dense = TimeDistributed(Decoder_Dense, name='reconstruction_layer')
    decoder_outputs = Decoder_Time_Dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    if verbose: model.summary()
    
    if weights is not None:
        model.load_weights(weights)
    model.compile(optimizer='adam', loss = 'categorical_crossentropy')

    # Define sampling models
    #Encoders
    encoder_model = Model(encoder_inputs, z_mean_tanh)
    encoder_model_sampling = Model(encoder_inputs, sampled_encoder_states)
    
    #Transform and decoder
    decoder_states_inputs = Input(shape=(latent_dim,))
    transformed_states = Decoder_Dense_Initial(decoder_states_inputs)
    transform_model = Model(decoder_states_inputs, transformed_states)

    transformed_states_inputs = Input(shape=(sum([256,256,256]),))
    decoder_cell_inital = spliter(transformed_states_inputs)
    decoder_outputs_and_states = decoder_gru(decoder_inputs, initial_state=decoder_cell_inital)
    decoder_states = decoder_outputs_and_states[1:]
    decoder_states = Concatenate(axis=1)(decoder_states)
    decoder_outputs = decoder_outputs_and_states[0]
    decoder_outputs = Decoder_Time_Dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + [transformed_states_inputs],
                          [decoder_outputs] + [decoder_states])

    return model, encoder_model, transform_model, decoder_model


class GANwCondition():
    def __init__(self, latent_dim, noise_dim, lr_g = 0.00005, lr_d= 0.00005, condition_dim=1449, verbose=True):       
        self.condition_dim = condition_dim
        self.latent_condition_size = 256
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim

        optimizer_g = RMSprop(lr_g)
        optimizer_c = RMSprop(lr_d)

        # Build  condition encoder
        self.condition_encoder = self.build_condition_encoder(verbose=verbose)
        
        # Build  and compile classifier 
        self.classifier = self.build_classifier(self.condition_encoder, verbose=verbose)
        self.classifier.compile(loss='binary_crossentropy', optimizer=optimizer_c, metrics=['accuracy'])
        
        # Build  generator for pahse 1
        self.G = self.build_generator(self.condition_encoder, name='Generator Phase 1', verbose=verbose)
               
        # Build  discriminator
        self.D = self.build_discriminator(name='Discriminator', verbose=verbose)
        
        # Build and compile the critic
        self.C = self.build_critic(self.D, name='Critic', verbose=verbose)
        self.C.compile(loss=[self.mean_loss, 'MSE'], loss_weights=[1, 10], optimizer=optimizer_c)

        # Build and compile StackGAN
        self.classifier_weight = K.variable(0.)
        self.GAN = self.build_GAN([self.G], self.D, self.classifier, inputSize=self.noise_dim, name='GAN', verbose=verbose)
        self.GAN.compile(loss=[self.wasserstein_loss, 'binary_crossentropy'], loss_weights=[1,5], optimizer=optimizer_g)
    
    def mean_loss(self, y_true, y_pred):
        return K.mean(y_pred)
    
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    
    def build_GAN(self, generators, discriminator, classifier, inputSize, name='Combinded Model', verbose=True):
        # For the combined model we will only train the generators
        discriminator.trainable = False
        classifier.trainable = False
        
        # The generator takes noise and the target label as input
        # and generates the corresponding latente space region with that condition
        noise = h = Input(shape=(inputSize,))
        condition = Input(shape=(self.condition_dim,))
        
        #c_mean  = condition_encoder(condition)
        #condition_sampled, mean_log_var = self.condition_sampler([c_mean, c_log_var])
        
        states = []
        for G in generators:
            h = G([h, condition])
            states.append(h)
        
        # The discriminator takes generated latent space cordinates as input and 
        # determines validity if they correspond to the condition
        valid_unconditioned = []
        for s in states:
            h = discriminator(s)
            valid_unconditioned.append(h)
        
        valid_conditioned = []
        for s in states:
            h = classifier([s, condition])
            valid_conditioned.append(h)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        combined = Model([noise, condition], valid_unconditioned + valid_conditioned)
        if verbose: logging.info('\n' + str(name))
        if verbose: combined.summary()
        
        return combined
    
    
    def build_critic(self, discriminator, name='Critic Model', verbose=True):
        
        real_states = Input(shape=(self.latent_dim,))
        fake_states = Input(shape=(self.latent_dim,))
                
        #########
        # Construct weighted average between real and fake images
        inter_states = Lambda(self.RandomWeightedAverage, name='RandomWeightedAverage')([real_states, fake_states])
        
        # The discriminator takes generated latent space cordinates as input and 
        # determines validity if they correspond to the condition
          
        valid = discriminator(real_states)
        fake = discriminator(fake_states)
        inter = discriminator(inter_states)
                
        sub = Subtract()([fake, valid])
        norm = inter_states = Lambda(self.GradNorm, name='GradNorm')([inter, inter_states])
        
        # output: D(G(Z))-D(X), norm ===(nones, ones)==> Loss: D(G(Z))-D(X)+lmbd*(norm-1)**2  
        critic_model = Model(inputs=[real_states, fake_states], outputs=[sub, norm])
        if verbose: logging.info('\n' + str(name))
        if verbose: critic_model.summary()
        
        return critic_model
    
    
    def build_condition_encoder(self, name='Condition Encoder', verbose=True):

        condition = Input(shape=(self.condition_dim,))
        
        #########

        h_condition = Dense(1024, activation=LeakyReLU(alpha=0.2))(condition)
        h_condition = Dense(512, activation=LeakyReLU(alpha=0.2))(h_condition)
        h_condition = Dense(256, activation=LeakyReLU(alpha=0.2))(h_condition)
        
        #########
        
        condition_encoder = Model(condition, h_condition)

        if verbose: logging.info('\n' + str(name))
        if verbose: condition_encoder.summary()
        
        return (condition_encoder)
    
    
    def build_generator(self, condition_encoder, name='Generator', verbose=True):

        noise = Input(shape=(self.noise_dim,))
        condition = Input(shape=(self.condition_dim,))
        condition_encoder.trainable = False
        
        #########
        
        h_nosie = Dense(512, activation=LeakyReLU(alpha=0.2))(noise)
        h_nosie = Dense(256, activation=LeakyReLU(alpha=0.2))(h_nosie)
        
        #########
        
        h_condition = condition_encoder(condition)
        
        #########
        
        h = concatenate([h_nosie, h_condition])
        h = Dense(256, activation=LeakyReLU(alpha=0.2))(h)
        h = Dense(self.latent_dim, activation='tanh')(h)
        
        #########
        
        generator = Model([noise, condition], h)

        if verbose: logging.info('\n' + str(name))
        if verbose: generator.summary()
        
        return (generator)
    
    
    def build_discriminator(self, name='Discriminator', verbose=True):
        
        states = Input(shape=(self.latent_dim,))
        
        ###########################
        # NO BATCH NORMALIZATION! #
        ###########################
        
        h = Dense(256, activation=LeakyReLU(alpha=0.2))(states)
        h = Dense(256, activation=LeakyReLU(alpha=0.2))(h)
        h = Dropout(rate=0.4)(h)
        
        #########
        
        h_unconditioned = Dense(256, activation=LeakyReLU(alpha=0.2))(h)
        h_unconditioned = Dropout(rate=0.4)(h_unconditioned)
        h_unconditioned = Dense(1, activation=None)(h_unconditioned)
        
        #########
        
        discriminator = Model(states, h_unconditioned)

        if verbose: logging.info('\n' + str(name))
        if verbose: discriminator.summary()

        return discriminator


    def build_classifier(self, condition_encoder, name='Classifier', verbose=True):
        
        states = Input(shape=(self.latent_dim,))
        condition = Input(shape=(self.condition_dim,))
        
        #########
        
        h = Dense(256, activation=LeakyReLU(alpha=0.2))(states)
        h = Dense(256, activation=LeakyReLU(alpha=0.2))(h)
        h = Dropout(rate=0.4)(h)
        
        h_condition = condition_encoder(condition)
        h_condition = Dropout(rate=0.4)(h_condition)
        
        #########

        h_conditioned = concatenate([h, h_condition])
        h_conditioned = Dense(256, activation=LeakyReLU(alpha=0.2))(h_conditioned)
        h_conditioned = Dropout(rate=0.4)(h_conditioned)
        h_conditioned = Dense(1, activation='sigmoid')(h_conditioned)
        
        #########
        
        classifier = Model([states, condition], h_conditioned)

        if verbose: logging.info('\n' + str(name))
        if verbose: classifier.summary()

        return classifier

    def RandomWeightedAverage(self, inputs):
            """Provides a (random) weighted average between real and generated image samples"""
            alpha = K.random_uniform(K.shape(inputs[0]))
            return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
        
    def GradNorm(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        if not len(grads) == 1:
            print("gradient must have length 1")
            exit()
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))
    
    def train(self, data, conditions, epochs, n_critic, batch_size, save_interval=500, save_to='results', verbose=True):
        n_epoch = save_interval
        self.losses = []
        self.FD = []
        data = np.asarray(data)
        conditions = np.asarray(conditions)
        idx = list(range(len(data)))
        
        for epoch in range(epochs):
            if verbose: logging.info('Epoch: '+str(epoch+1)+'/'+str(epochs))
            
            np.random.shuffle(idx)
            steps = int(math.ceil(len(conditions) / float(batch_size)))
                        
            for step in range(steps):
                
                # Select a random half batch of molecules
                idx_step = idx[step*batch_size:(step+1)*batch_size]
                samples = len(idx_step)
                
                real_data_step = data[idx_step]
                real_conditions_step = conditions[idx_step]
                noise = np.random.normal(0, 1, (samples, self.noise_dim))
                
                # Generate a half batch of new data               
                G1_out = self.G.predict([noise, real_conditions_step])

                # Generate labels
                valid = np.ones((samples, 1))
                fake = np.zeros((samples, 1))
                
                # ----------------------
                #  Train Critics
                # ----------------------
                
                # Train the C1
                c_loss = self.C.train_on_batch([real_data_step, G1_out], [valid, valid])
                                
                # ---------------------
                #  Train Generators
                # ---------------------
                if (step % n_critic == 0 or step+1 == steps):
                    noise = np.random.normal(0, 1, (samples, self.noise_dim))
                
                    # Train both generators
                    g_loss = self.GAN.train_on_batch([noise, real_conditions_step], [valid*-1, valid])
                
                    if verbose: logging.info("%d/%d [D loss: %f] [G1 loss: %f] [G1 class: %f]" %
                                       (step, steps, c_loss[0], g_loss[1], g_loss[2]))
                    self.losses.append(c_loss[0:1]+g_loss)
         
            # Calculate Frechet Distance and classification loss
            noise = np.random.normal(0, 1, (len(conditions), self.noise_dim))
                            
            G1_out = self.G.predict([noise, conditions])
            class_loss = self.classifier.evaluate([G1_out, conditions], np.ones((len(conditions), 1)), verbose = 0)
            
            FD1 = self.calculate_frechet_distance(mu1=np.mean(G1_out, axis=0), mu2=np.mean(data, axis=0),
                                                  sigma1=np.cov(G1_out.T), sigma2=np.cov(data.T))
            if verbose: 
                logging.info("\nEpoch: %d/%d [Frechet Distance 1: %f] [Class 1: %f]\n" %((epoch+1), epochs, FD1, class_loss[0]))
            self.FD.append((FD1, class_loss[0]))                         
            
            # Save results
            if save_interval and ((epoch+1) % n_epoch == 0 or epoch+1 == epochs):
                checkdir = os.path.join(os.getcwd(),save_to)
                if not os.path.exists(checkdir):
                    os.makedirs(checkdir)#, exist_ok=True)
                    
                self.C.save_weights(os.path.join(checkdir, "gan_C_" + str(epoch+1) + "epochs.h5"))
                self.D.save_weights(os.path.join(checkdir, "gan_D_" + str(epoch+1) + "epochs.h5"))
                self.G.save_weights(os.path.join(checkdir, "gan_G_" + str(epoch+1) + "epochs.h5"))
                self.condition_encoder.save_weights(os.path.join(checkdir, "gan_condition_encoder_" + str(epoch+1) + "epochs.h5"))
                if verbose: logging.info('Saving to', checkdir, '\n')
               
        return (self.losses, self.FD)
    
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).            
        Stable version by Dougal J. Sutherland.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        if not mu1.shape == mu2.shape:
            print("Training and test mean vectors have same shapes")
            exit()

        if not sigma1.shape == sigma2.shape:
            print("Training and test covariances must have same shapes")
            exit()

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        

