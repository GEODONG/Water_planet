from hefesto_analysis import Simulation

sim = Simulation(name="test", title="Test Run", dir="/path/to/data")
sim.read_profile()
sim.plot_Mg_Si_ratio_average_all()
