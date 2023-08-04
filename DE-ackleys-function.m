%-------------------------------------------------------------------------------
% Group 1 - Implementation of differential evolution to solve ackelys function
% index - AS2019566, AS2019574 ,AS2019358, AS2019554
%-------------------------------------------------------------------------------

%------------------------------------------------
% seperate functions -> mutation, crossover
%          setting boundries, selection,
%          plot the results
%------------------------------------------------

%-----------------------------------------
%clean the environment
%-----------------------------------------
clc;  
clear;

%-----------------------------------------
%Initilized the parameters
%-----------------------------------------
num_params = 2;          % Number of parameters to optimize
num_dimensions = 2;      % Number of dimensions (2 for 2D optimization)
population_size = 10;    % Size of the population
F = 0.5;                 % scaling factor
CR = 0.1;                % crossover rate
lower_bound = -5 * ones(1, 2);     % Lower bound of parameter values
upper_bound = 5 * ones(1, 2);      % upper bound


% Define the no of iterations and no of runs

runs=5;
iteration = 50;

%-----------------------------------------
%  Function - ackleys function   
%  objective function
%----------------------------------------- 

function fitness = evaluate_ackleys_function(vector)
  
  x1 = vector(1);
  x2 = vector(2);
  term1 = -0.2 * sqrt(0.5 * (x1^2 + x2^2));
  term2 = 0.5 * (cos(2 * pi * x1) + cos(2 * pi * x2));
  fitness = 20 + exp(1) - 20 * exp(term1) - exp(term2);
end

%------------------------------------------------------------
% DE -> initialization -> mutation -> crossover -> selection
%------------------------------------------------------------

%---------------------------------------------------------
% Function - Mutation function
% parameters - Scalling factor , population , ith value
%---------------------------------------------------------

function mutant = mutation ( i, F,population)
  
  % Choose three unique individuals (a, b, c) randomly , the selectd 3 individuals not equal to current individual value
  idxs = 1:size(population, 1);
  idxs(i) = []; % Exclude the current individual from selection
  selected_indices = idxs(randperm(length(idxs), 3));
  a = population(selected_indices(1), :);
  b = population(selected_indices(2), :);
  c = population(selected_indices(3), :);
  
  % Perform mutation 
  fprintf('mutant \n')
  mutant = a + F * (b - c);
  
end

%---------------------------------------------------------
% Function - Crossover function
% parameters - population, mutation vector (donor vector), 
%              Crossover Rate , no of dimentions
%---------------------------------------------------------

function crossover_vector = crossover(problem, mutant, CR,num_dimensions)
  
  % Generate a random index for crossover
  crossover_point = randi(1,num_dimensions);
  
  % Initialize the crossover vector with the target vector
  crossover_vector = mutant;
  
  % Perform binomial crossover for each element
  for i = 1:2
    if rand <= CR || i == crossover_point
      % If the random value is less than CR or it is the crossover point,
      % use the trial vector element
      crossover_vector(i) = problem(i);
    end
  end
end

%-------------------------------------------------------------
% Function - Setting Boundries
% Use - ensure that trial vector(crossover_vector) within the 
%       feasible region during the optimization process           
% parameters - population, mutation vector (donor vector), 
%              Crossover Rate , no of dimentions
%-------------------------------------------------------------

function crossover_vector= boundries (crossover_vector,lower_bound,upper_bound,i) 
  for i=1:2 
    if crossover_vector(i) < lower_bound(i)
      %set the boundries
      crossover_vector(i) = lower_bound(i);
    elseif crossover_vector(i) > upper_bound
      crossover_vector(i) = upper_bound(i);
    end
  end
end

%---------------------------------------------------------
% Function - Selection         
% parameters - population, crossover_vector
%---------------------------------------------------------

function selected_vector = selection(problem, crossover_vector)
  disp('selection');
  
  % get the population fitness values
  target_fitness = evaluate_ackleys_function(problem);
  
  % get the donor vector fiteness values
  crossover_fitness = evaluate_ackleys_function(crossover_vector);
  
  % check whether wich one in minimun
  if crossover_fitness < target_fitness
    %choose minimum value as selected vector
    selected_vector = crossover_vector;
  else
    selected_vector = problem;
  end
end


%---------------------------------------------------------
% Function - Best Fitness Value (fitness evaluation)
% Use - find the best solution and best fitness value   
%       using ackleys function       
% parameters - population, population_size, 
%              No of runs
%---------------------------------------------------------
function [update_best_fitness,best_solutions] = bestfitness(population,population_size,runs)
  
  % assign the big value to the update_best_fitness variable
  update_best_fitness = inf;
  
  % Evaluate the fitness of the current population
  for i = 1: population_size
    disp(population(i,:));
    population_fitness = evaluate_ackleys_function(population(i,:))
    
    % Update the best fitness value if a new best value is found
    current_best_fitness = min(population_fitness);
    if current_best_fitness < update_best_fitness
      update_best_fitness = current_best_fitness;
      best_solutions = population(i,:)
    end
  end
  
  %print the results
  fprintf('Best firness value for run %d : ',runs );
  disp(update_best_fitness);
end

%-------------------------------------------------------------
% plot the results 
% 1. Best Fitness Values vs. Iterations for a particular run
% 2. Best Fitness Values vs. Runs
% 3. Running time vs. Runs
%-------------------------------------------------------------

function plot_best_fitness_values_against_iteration(iteration_value_bestfitness)
  
  plot(iteration_value_bestfitness(:,1), iteration_value_bestfitness(:,2), 'o-');
  xlabel('Iteration');
  ylabel('Best Fitness Value');
  title('Best Fitness Values vs. Iterations');
  ylim([min(iteration_value_bestfitness(:,2)-1), max(iteration_value_bestfitness(:,2))+1]);
  
endfunction

function plot_best_fitness_values_against_runs(best_fitness_values)
  plot(best_fitness_values(:,1), best_fitness_values(:,2), 'o-');
  xlabel('Runs');
  ylabel('Best Fitness Value');
  title('Best Fitness Values vs. Runs');
  ylim([min(best_fitness_values(:,2)-1), max(best_fitness_values(:,2))+1]);
endfunction

function plot_running_time_against_run(running_time_per_run)
  plot(running_time_per_run(:,1), running_time_per_run(:,2), 'o-');
  xlabel('Run');
  ylabel('Running time');
  title('Running time vs. Runs');
  ylim([min(running_time_per_run(:,2)-1), max(running_time_per_run(:,2))+1]);
endfunction

%---------------------------------------------------------
% Function - plot the ackleys function in mesh grid        
%---------------------------------------------------------

function plot_ackleys_function()
  
  % Define the range of x and y values
  x_range = linspace(-5, 5, 100);
  y_range = linspace(-5, 5, 100);
  %x_range = population(:,1)
  %y_range = population(:,2)
  
  % Initialize a matrix to store the function values
  Z = zeros(length(x_range), length(y_range));
  
  % Calculate the values of Ackley's function for each combination of x and y
  for i = 1:length(x_range)
    for j = 1:length(y_range)
      Z(i, j) = evaluate_ackleys_function([x_range(i), y_range(j)]);
    end
  end
  
  % Create the surface plot
  figure;
  surf(x_range, y_range, Z, 'EdgeColor', 'none');
  xlabel('X');
  ylabel('Y');
  zlabel('Ackley''s Function Value');
  title('Ackley''s Function');
  colormap('jet');
  colorbar;
  grid on;
  
  %Find the global minimum (0, 0) and plot it
  % hold on;
  % plot3(0, 0, evaluate_ackleys_function([0, 0]), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
  % hold off;
  
end

%----------------------------------------------------------------
% Function - Main function 
% Use - to run algorithm 
% parameters - no of parameters, no of dimensions, population_size
%              scalling factor , crossover rate, lower bound,
%              upper bound, runs, iterations
%----------------------------------------------------------------

function DE_ackleys_function(num_params,num_dimensions,population_size,F,CR,lower_bound,upper_bound,runs,iteration)
  
  % initialization
  best_fitness_values = [];
  best_fitness_solutions = [];
  iteration_value_bestfitness = [];
  random_run = randi([1,runs])
  running_time_per_run =[];
  
  for run = 1:runs 
    tic; % start running time 
    disp('---------run--------');
    fprintf('Run number %d \n',run);
    
    %define initial population randomly
    population = repmat(lower_bound,population_size,1) + repmat((upper_bound - lower_bound),population_size,1) .* rand(population_size, num_params);
    
    disp(population);
    
    %initialization
    best_fitness_value_iteration = 0;
    best_solutions_for_iteration = NaN;
    
    for iterations = 1:iteration
      %initialization
      new_population = zeros(size(population));
      
      for i = 1:population_size
        
        %mutation
        mutant = mutation ( i, F,population);
        
        %crossover
        crossover_vector = crossover(population(i, :), mutant, CR,num_dimensions);    
        
      endfor 
      
      for i = 1:population_size
        
        % Apply boundary handling 
        crossover_vector= boundries (crossover_vector,lower_bound,upper_bound,i);
        
        %selection 
        new_population(i, :) = selection(population(i, :), crossover_vector);
        
      endfor
      
      % Update the population for a one generation
      population = new_population;
      
      %disp(population);
      
      % fitness evaluate for iteration
      disp('update best fiteness');
      [update_best_fitness,best_solutions] = bestfitness(new_population,population_size,iterations);
      best_fitness_value_iteration = update_best_fitness
      best_solutions_for_iteration = best_solutions
      
      if run == random_run
        iteration_value_bestfitness = [iteration_value_bestfitness;[iterations,update_best_fitness]]
      endif
      
    endfor
    %fitness evaluate for run
    best_fitness_values = [best_fitness_values;[run,best_fitness_value_iteration]]
    best_fitness_solutions=[best_fitness_solutions;best_solutions_for_iteration]
    
    %stop running time
    running_time_per_run = [running_time_per_run;[run,toc]];
  end
  
  %plot results
  
  plot_ackleys_function(); 
  hold on;
  plot3(best_fitness_solutions(1), best_fitness_solutions(2), evaluate_ackleys_function([best_fitness_solutions(1), best_fitness_solutions(2)]), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
  hold off;
  
  figure;
  plot_best_fitness_values_against_iteration(iteration_value_bestfitness);
  figure;
  plot_best_fitness_values_against_runs(best_fitness_values);
  figure;
  plot_running_time_against_run(running_time_per_run);
  
  
end
% Call the main function
DE_ackleys_function(num_params,num_dimensions,population_size,F,CR,lower_bound,upper_bound,runs,iteration);