Synthetic Control With Multiple Outcomes
=====================

.. autoclass:: mlsynth.mlsynth.SCMO
   :show-inheritance:
   :special-members: __init__


Uses SC with multiple outcomes in a simulated setting.


.. code-block:: python

  import numpy as np
  import pandas as pd
  from mlsynth.mlsynth import SCMO
  
  def simulate(
      N=99, T0=52*3, T1=52, K=4, r=2, sigma=.20,
      max_gamma=0, T_season=12, seed=2000
  ):
      np.random.seed(seed)
      T = T0 + T1
  
      # Latent factors
      phi = np.random.normal(0, 1, size=(N, r))           # Market-specific latent factors (loadings)
      mu = np.random.normal(0, 1, size=(T, K, r))          # Time-and-outcome-specific latent factors
  
      # Fixed effects
      alpha = np.random.normal(0, 1, size=(N, K))          # Market fixed effects
      beta = np.random.normal(0, 1, size=(T, K))           # Time fixed effects
  
      # Market-specific seasonal parameters
      gamma_i = np.random.uniform(0, max_gamma, size=N)    # amplitude of seasonal effect
      tau_i = np.random.randint(0, T_season, size=N)       # phase shift (peak week)
  
      # Construct seasonal matrix S (N x T): market-time-specific seasonality
      t_grid = np.arange(T)
      S = np.array([
          gamma_i[i] * np.cos(4 * np.pi * (t_grid - tau_i[i]) / T_season)
          for i in range(N)
      ])
  
      # Outcome tensor
      Y = np.zeros((N, T, K))
  
      # Base shift
      baseline_shift = [np.random.randint(200, 500) for _ in range(K)]  # Random base values for each outcome
  
      # Autocorrelation coefficients for each outcome
      rho = np.array([0.8, 0.6, 0.5, 0.3])  # AR(1) coefficients for each outcome, can we adjusted if we wish.
  
      for k in range(K):
          latent = phi @ mu[:, k, :].T  # N x T
          base = (
              alpha[:, [k]] +         # N x 1,
              beta[:, k] +            # T,
              latent +                # N x T,
              S +                     # N x T
              baseline_shift[k]       # scalar
          )
          noise = np.random.normal(0, sigma, size=(N, T))
  
          # First time point initialization
          Y[:, 0, k] = base[:, 0] + noise[:, 0]
  
          # Autoregressive Factors
          for t in range(1, T):
              Y[:, t, k] = (
                  rho[k] * Y[:, t-1, k] +
                  (1 - rho[k]) * base[:, t] +
                  noise[:, t]
              )
  
      # Identify treated market: second-highest factor loading
      treated_unit = np.argsort(phi[:, 0])[-2]
  
      time = np.arange(T)
      post_treatment = (time >= T0)
      treat = np.zeros((N, T), dtype=int)
      treat[treated_unit, post_treatment] = 1
  
      # Inject treatment effect into Gross Booking Value for treated market
      Y[treated_unit, post_treatment, 0] += 5  # add treatment effect of +5, but can be whatever we like.
  
      # Construct the dataframe without loops
      markets = np.arange(N)[:, None]           # shape (N, 1)
      weeks = np.arange(T)[None, :]             # shape (1, T)
  
      market_grid = np.repeat(markets, T, axis=1).flatten()  # shape (N*T,)
      week_grid = np.tile(weeks, (N, 1)).flatten()           # shape (N*T,)
  
  
      cities = [
          "São Paulo", "Mexico City", "San Carlos de Bariloche", "Rio de Janeiro", "Ushuaia",
          "Bogotá", "Santiago", "Caracas", "Guayaquil", "Quito",
          "Brasília", "Bocas del Toro", "Asunción", "Cabo San Lucas", "Playa del Carmen",
          "Medellín", "Porto Alegre", "Placencia", "Recife", "Salvador",
          "Zihuatanejo", "San José", "Panama City", "Montevidio", "Tegucigalpa",
          "Foz do Iguaçu", "Maracaibo", "Rosario", "Maracay", "Antofagasta",
          "San Pedro Sula", "San Juan", "Chihuahua", "Cayo District", "Maturín",
          "Buzios", "Puebla", "Mar del Plata", "Arequipa", "Fernando de Noronha", "Guatemala City",
          "Mazatlán", "Mérida", "Córdoba", "Cozumel", "Trujillo",
          "Corozal Town", "Santa Cruz de la Sierra", "San Luis Potosí", "Jalapão", "Potosí",
          "Tucumán", "Neuquén", "La Plata", "Viña del Mar", "Florianópolis", "Lagos de Moreno",
          "La Paz", "Belém", "Venezuela", "Ribeirão Preto", "Valparaíso",
          "Marília", "Campinas", "Vitoria", "Sorocaba", "Santa Fe",
          "San Salvador", "Lima", "Buenos Aires", "Curitiba", "Maceió",
          "Cartagena", "La Ceiba", "Puerto La Cruz", "Olinda", "Monterrey",
          "Ibagué", "Cúcuta", "Playa Venao", "Cancún", "Puerto Escondido", "Chiclayo", "Ambato",
          "Pucallpa", "Santa Marta", "Villavicencio", "Paraná", "Cauca", "San Vicente",
          "Cali", "Tarija", "Manzanillo", "El Alto", "Santiago de Chile", "Cochabamba",
          "Punta del Este", "Iquique",  "Durango", "Puerto Viejo de Talamanca"
      ]
  
      city_mapping = {i: cities[i] for i in range(N)}
  
      data = {
          'Market': [city_mapping[market] for market in market_grid],
          'Week': week_grid,
          'Experiences': treat.flatten()
      }
  
      for k in range(K):
          if k == 0:
              data['Gross Booking Value'] = Y[:, :, k].flatten()
          elif k == 1:
              data['Average Booking Price'] = Y[:, :, k].flatten()
          elif k == 2:
              data['Average Daily Visitors'] = Y[:, :, k].flatten()
          elif k == 3:
              data['Average Cost of Hotel Rooms'] = Y[:, :, k].flatten()
  
      return pd.DataFrame(data)
  
  # Run simulation
  df = simulate(seed=10000, r=3)
  config = {
      "df": df,
      "outcome": 'Gross Booking Value',
      "treat": 'Experiences',
      "unitid": 'Market',
      "time": 'Week',
      "display_graphs": True,
      "save": False,
      "counterfactual_color": ["blue"], "addout": list(df.columns[4:]),
      "method": "both"
  }
  
  arco = SCMO(config).fit()
