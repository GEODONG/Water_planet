import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16



# Absolute path to the HeFESTo directory of the datat: /projects/JIEDENG/Donghao/Group_Tutorial/HeFESTo

class Simulation:
    def __init__(self, name, title, dir):
        self.name = name
        self.title = title
        self.dir = dir
        self.profile_data = os.path.join(self.dir, 'fort.56')
        self.phase_density_data = os.path.join(self.dir, 'fort.61')
        self.atomicfraction_data = os.path.join(self.dir, 'fort.66')
        self.volfraction_data = os.path.join(self.dir, 'fort.67')
        self.all_output_data = os.path.join(self.dir, 'qout')
        self.phase_tables = {}        # {section_idx: DataFrame}
        self._build_tables()
        self.fraction = {}
        self.phase_cols = []
        
    def read_profile(self, save=False):
        """
        Read `fort.56`, skipping the banner line (skiprows=1),
        convert all numeric columns to float, apply unit factors,
        and keep only tidy, canonical names.
        """
        df = pd.read_csv(
            self.profile_data,
            delim_whitespace=True,
            skiprows=1,          # header actually starts on the 2nd physical line
            comment='#'          # just in case there are comment lines later
        )

        rename_map = {
            r'^P\(GPa\)$'          : ('P',      1.0),
            r'^depth.*\(km\)$'     : ('depth',  1.0),
            r'^T\(K\)$'            : ('T',      1.0),
            r'^rho\(g/cm\^3\)$'    : ('rho',    1.0),
            r'^VP\(km/s\)$'        : ('VP',     1.0),
            r'^VS\(km/s\)$'        : ('VS',     1.0),
            r'^VB\(km/s\)$'        : ('VB',     1.0),
            r'^VPQ\(km/s\)$'       : ('VPQ',    1.0),
            r'^VSQ\(km/s\)$'       : ('VSQ',    1.0),
            r'^H\(kJ/g\)$'         : ('H',      1.0),
            r'^S\(J/g/K\)$'        : ('S',      1.0),
            r'^alpha\(1e5_K\^-1\)$': ('alpha',  1.0),
            r'^cp\(J/g/K\)$'       : ('cp',     1.0),
            r'^KS\(GPa\)$'         : ('KS',     1.0),
            r'^Qs\(-\)$'           : ('Qs',     1.0),
            r'^Qp\(-\)$'           : ('Qp',     1.0),
            r'^rho_0\(g/cm\^3\)$'  : ('rho_0',  1.0),
        }

        original_cols = []
        for pattern, (new, factor) in rename_map.items():
            matches = df.filter(regex=pattern).columns
            if not len(matches):
                continue
            col = matches[0]

            # --- make sure values are numeric ---
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # --- apply unit conversion ---
            df[new] = df[col] * factor
            original_cols.append(col)

        # Drop the verbose original columns we just converted
        df.drop(columns=original_cols, inplace=True)

        self.profile = df
        
        # Save the profile data to a CSV file
        if save:
            output_file = os.path.join('./Real_run_test/Output_data', 'profile_data'+self.name+'.csv')
            df.to_csv(output_file, index=False)
            print(f"Profile data saved to {output_file}")

    def read_phase_fraction(self,save=False, fraction_type='atomic'):
        if fraction_type == 'atomic':
            df = pd.read_csv(self.atomicfraction_data, delim_whitespace=True, comment='#')
        elif fraction_type == 'volume':
            df = pd.read_csv(self.volfraction_data, delim_whitespace=True, comment='#')

        # Identify P, depth, T by regex (handles Pi/pi, Ti/ti)
        P_col  = df.filter(regex=r'^P[iI]$').columns[0]
        z_col  = df.filter(regex=r'^depth$').columns[0]
        T_col  = df.filter(regex=r'^T[iI]$').columns[0]

        df['P']     = df[P_col]
        df['depth'] = df[z_col] 
        df['T']     = df[T_col]

        # Phase columns = everything after the first three
        core = {P_col, z_col, T_col}
        phase_cols = [c for c in df.columns if c not in core]

        # Make sure they are floats (they might come in as str)
        df[phase_cols] = df[phase_cols].apply(pd.to_numeric, errors='coerce')

        # Calculate dominant phase row-by-row
        df['dominant_phase'] = df[phase_cols].idxmax(axis=1)

        self.fraction = df[['depth'] + phase_cols]
        
        # Save the volume fraction data to a CSV file
        if save:
            output_file = os.path.join('./Real_run_test/Output_data', fraction_type+'fraction_data'+self.name+'.csv')
            df.to_csv(output_file, index=False)
            print(fraction_type + f" fraction data saved to {output_file}")
            
    def read_phase_density(self, save=False):
        """
        Read `fort.61`, skipping the banner line (skiprows=1),
        convert all numeric columns to float, apply unit factors,
        and keep only tidy, canonical names.
        """
        df = pd.read_csv(
            self.phase_density_data,
            delim_whitespace=True,
            skiprows=0,          # header actually starts on the 2nd physical line
            comment='#'          # just in case there are comment lines later
        )

        # header is like:  Pi    depth       Ti     rhcpx       rhopx       rhol        ..., the name of the first three columns are fixed but the rest are variable
        P_col  = df.filter(regex=r'^P[iI]$').columns[0]
        z_col  = df.filter(regex=r'^depth$').columns[0]
        T_col  = df.filter(regex=r'^T[iI]$').columns[0]

        df['P']     = df[P_col]
        df['depth'] = df[z_col] 
        df['T']     = df[T_col]

        # Phase columns = everything after the first three
        core = {P_col, z_col, T_col}
        phase_cols = [c for c in df.columns if c not in core]
        
        # rename the phase columns to tidy names by removing the 'rh' prefix
        phase_cols = [col.replace('rh', '') for col in phase_cols]
        df.rename(columns=dict(zip(df.columns[3:], phase_cols)), inplace=True)
        
        # Make sure they are floats (they might come in as str)
        df[phase_cols] = df[phase_cols].apply(pd.to_numeric, errors='coerce')
        
        # Drop the verbose original columns we just converted
        #df.drop(columns=original_cols, inplace=True)

        self.phase_density = df
        
        #display the first few rows of the species density data
        #print(self.phase_density.head())
        
        # Save the species density data to a CSV file
        if save:
            output_file = os.path.join('./Real_run_test/Output_data', 'species_density_data'+self.name+'.csv')
            df.to_csv(output_file, index=False)
            print(f"Species density data saved to {output_file}")
    # The section below can locate the specific number in the table       
    # Get lines between the start and end lines for each index, each section is a table
    # Header of each table is           Si          Mg          Fe          Ca          Al          Na          Cr          O          XFe        Fe3+
    # where the first column is the phase name, which is not explicitly listed in the header
    # If I want to locate one specific number in the table, I need to know the index, the proper element name (which column), and the phase name in the first column (which row)
    # Design the following function and data structure to store the data
    def get_phase_chemistry(self, section_idx, phase_name, element_name):
        """
        Return the value (float) located at the intersection of
          • the 'phase_name' row
          • the 'element_name' column
          • the section # section_idx   (0‐based)
        """
        try:
            df = self.phase_tables[section_idx]
        except KeyError:
            raise ValueError(f"Section {section_idx} not found. "
                             f"Valid sections: {list(self.phase_tables)}") from None

        if phase_name not in df.index:
            raise ValueError(f"Phase '{phase_name}' not in section {section_idx}. "
                             f"Choices: {list(df.index)}")

        if element_name not in df.columns:
            raise ValueError(f"Element '{element_name}' not in section {section_idx}. "
                             f"Choices: {list(df.columns)}")

        return df.loc[phase_name, element_name] 
       
        
    def _build_tables(self):
        with open(self.all_output_data, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # 1. find each table’s start & stop
        starts = [i for i, ln in enumerate(lines) if "Phase Compositions - Cations" in ln]
        ends   = [i for i, ln in enumerate(lines) if "Phase Compositions - Mass Standard Oxides (%)" in ln]

        if len(starts) != len(ends):
            raise RuntimeError("Unequal number of start and end markers. "
                               "Check the file structure.")
            
        self.num_sections = len(starts)

        # 2. loop over every section
        for idx, (s, e) in enumerate(zip(starts, ends)):
            raw_section = lines[s + 1 : e]                 # +1 skips the tag line itself
            table = self._parse_one_table(raw_section)     # DataFrame
            self.phase_tables[idx] = table
   
    @staticmethod
    def _parse_one_table(raw_lines):
        """
        Expects something like:

           Si   Mg   Fe   ... Fe3+
        Ol  0.  1.2  0.3  ... 0.04
        Px  2.  0.8  0.5  ... 0.01
        ...

        First non-blank line is treated as *header*.
        """
        # Remove blank lines and trailing newlines
        raw_lines = [ln.rstrip() for ln in raw_lines if ln.strip()]

        if not raw_lines:
            return pd.DataFrame()      # empty table; return blank DF

        # 1. header (everything after the first chunk of whitespace)
        header_tokens = raw_lines[0].split()
        columns = ['Phase'] + header_tokens

        # 2. data rows
        data = []
        
        for ln in raw_lines[1:]:
            parts = ln.split()
            if len(parts) < 2:         # skip malformed lines
                continue
            phase = parts[0]
            # convert everything that *looks* like a number; keep
            # non-numeric tokens (rare) as strings
            nums  = [float(x) if re.match(r'^[-+]?\d*\.?\d+(e[-+]?\d+)?$', x, re.I) else x
                     for x in parts[1:]]
            data.append([phase] + nums)

        df = pd.DataFrame(data, columns=columns).set_index('Phase')
        return df
        
    def calc_Mg_Si_ratio_single(self, section_idx):
        """
        Calculate the Mg/Si ratio for a given phase in a given section.
        """
        try:
            df = self.phase_tables[section_idx]
        except KeyError:
            raise ValueError(f"Section {section_idx} not found. "
                             f"Valid sections: {list(self.phase_tables)}") from None
        
        if phase_name not in df.index:
            raise ValueError(f"Phase '{phase_name}' not in section {section_idx}. "
                             f"Choices: {list(df.index)}")

        # Get the Mg and Si values for the specified phase
        Mg = df.loc[phase_name, 'Mg']
        Si = df.loc[phase_name, 'Si']

        # Calculate the Mg/Si ratio
        if Si == 0:
            return np.nan
        
    def calc_Mg_Si_ratio_average(self):
        """
        Calculate the Mg/Si ratio
        Equation:
            total Mg / total Si
        """
        fraction_idx_len = len(self.fraction.index)
        if fraction_idx_len != self.num_sections:
            raise ValueError(f"Number of sections in the fraction data ({fraction_idx_len}) in fort.67 file "
                             f"does not match the number of chemistry tables ({self.num_sections}) in qout file.")
        
        Mg_total_array = np.zeros(fraction_idx_len)
        Si_total_array = np.zeros(fraction_idx_len)
        for idx in range(fraction_idx_len):
            mineral_phases = self.phase_tables[idx].index
            Mg_total = 0
            Si_total = 0
            for phase_name in mineral_phases:
                try:
                    df = self.phase_tables[idx]
                except KeyError:
                    raise ValueError(f"Section {idx} not found. "
                                     f"Valid sections: {list(self.phase_tables)}") from None

                frac_i = self.fraction[phase_name].values[idx] 
                Mg_i = self.get_phase_chemistry(idx, phase_name, 'Mg')
                Si_i = self.get_phase_chemistry(idx, phase_name, 'Si')
                Mg_total += frac_i * Mg_i
                Si_total += frac_i * Si_i
            Mg_total_array[idx] = Mg_total
            Si_total_array[idx] = Si_total
        # Calculate the Mg/Si ratio
        self.Mg_Si_ratio = Mg_total_array / Si_total_array
        return self.Mg_Si_ratio
    
    def calc_Mg_Si_ratio_average_2(self):
        """
        Calculate Mg/Si ratio
        Equation:
            sum of (fraction * Mg/Si) for each phase
        """
        fraction_idx_len = len(self.fraction.index)
        if fraction_idx_len != self.num_sections:
            raise ValueError(f"Number of sections in the fraction data ({fraction_idx_len}) in fort.67 file "
                             f"does not match the number of chemistry tables ({self.num_sections}) in qout file.")
        
        Mg_total_array = np.zeros(fraction_idx_len)
        Si_total_array = np.zeros(fraction_idx_len)
        Mg_Si_ratio_array = np.zeros(fraction_idx_len)
        for idx in range(fraction_idx_len):
            mineral_phases = self.phase_tables[idx].index
            Mg_Si_ratio = 0
            for phase_name in mineral_phases:
                try:
                    df = self.phase_tables[idx]
                except KeyError:
                    raise ValueError(f"Section {idx} not found. "
                                     f"Valid sections: {list(self.phase_tables)}") from None

                frac_i = self.fraction[phase_name].values[idx] 
                Mg_i = self.get_phase_chemistry(idx, phase_name, 'Mg')
                Si_i = self.get_phase_chemistry(idx, phase_name, 'Si')
                if Mg_i == 0 or Si_i == 0:
                    Mg_Si_ratio_i = 0
                else:
                    Mg_Si_ratio_i = Mg_i / Si_i
                Mg_Si_ratio += frac_i * Mg_Si_ratio_i
            Mg_Si_ratio_array[idx] = Mg_Si_ratio
        # Calculate the Mg/Si ratio
        self.Mg_Si_ratio = Mg_Si_ratio_array
        
        return self.Mg_Si_ratio

    def plot_Mg_Si_ratio_average_all(self):
        """
        Plot the average Mg/Si ratio for all sections.
        Equation:
            total Mg / total Si
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.fraction['depth'], self.Mg_Si_ratio, linestyle='-', color='C3', label='Mg/Si Ratio')
        ax.set_ylabel('Mg/Si Ratio')
        ax.set_title('Average Mg/Si Ratio for All Sections')
        ax.legend()
        plt.show()
        
    def plot_Mg_Si_ratio_average_all_2(self):
        """
        Plot the average Mg/Si 
        Equation:
            sum of (fraction * Mg/Si) for each phase
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.fraction['depth'], self.Mg_Si_ratio, linestyle='-', color='C3', label='Mg/Si Ratio')
        ax.set_ylabel('Mg/Si Ratio')
        ax.set_title('Average Mg/Si Ratio for All Sections')
        ax.legend()
        plt.show()
        
    def mineral_fraction_vs_depth(self, contrast=False, pspl_boundary=False, more_info=False, ticks_type='pressure'):
        """
        Plot the mineral fraction vs depth for a specific mineral.
        """
        # select the phase that are not constantly 0
        phase_cols = [col for col in self.fraction.columns if col not in ['P', 'depth', 'T', 'dominant_phase']]
        self.phase_cols = [col for col in phase_cols if self.fraction[col].max() > 0.0]
        
        # get color from rainbow colormap for each phase
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i/len(self.phase_cols)) for i in range(len(self.phase_cols))]
        maximum_Depth = self.fraction['depth'].max()
        interval = self.fraction['depth'].max()/len(self.fraction['P'])
        
        
        # build color dictionary for each phase
         
        self.color_dict = {'plg': (0.9811764705882353, 0.6470588235294117, 0.6188235294117648),
        'sp': (0.9952941176470589, 0.8211764705882352, 0.5811764705882352),
        'opx': (0.76, 0.9058823529411764, 0.7270588235294118),
        'c2c': (0.9388235294117646, 0.9388235294117646, 0.9388235294117646),
        'cpx': (0.9811764705882353, 0.4023529411764705, 0.33647058823529413),
        'gt': (0.9905882352941178, 0.6470588235294117, 0.2611764705882353),
        'cpv': (0.9858823529411764, 0.7647058823529411, 0.8776470588235294),
        'ol': (0.6423529411764706, 0.7647058823529411, 0.868235294117647),
        'wa': (1.0, 1.0, 0.76),
        'ri':(0.6423529411764706, 0.8447058823529412, 0.2941176470588235),
        'il': (0.8823529411764706, 0.9529411764705882, 0.7458823529411764),
        'pv': (0.4635294117647059, 0.792941176470588, 0.736470588235294),
        'ppv': (0.4023529411764705, 0.6329411764705881, 0.792941176470588),
        'cf': (0.8211764705882352, 0.8211764705882352, 0.8211764705882352),
        'nal': (0.8447058823529412, 0.7552941176470587, 0.8729411764705883),
        'mw': (0.8776470588235294, 0.8164705882352941, 0.6894117647058824),
        'qtz': (0.9905882352941178, 0.7647058823529411, 0.6094117647058823),
        'st': (0.76, 0.76, 0.76),
        'ky': (1.0, 0.9388235294117646, 0.6188235294117648),
        'neph': (0.6423529411764706, 0.8635294117647059, 0.7647058823529411),
        'feg': (0.6847058823529413, 0.4023529411764705, 0.6894117647058824),
        'fee': (0.9482352941176471, 0.7505882352941176, 0.8729411764705883),
        'fea': (0.9905882352941178, 0.8258823529411765, 0.9105882352941177),
        'apbo': (0.7552941176470587, 0.8023529411764707, 0.891764705882353),
        'coes': (0.9341176470588235, 0.8635294117647059, 0.76),
        'pwo': (0.6941176470588235, 0.6752941176470588, 0.8258823529411765),
        'wo': (1.0, 1.0, 0.6423529411764706),
        'pspl': (0.9905882352941178, 0.8258823529411765, 0.9105882352941177),}
        
        if more_info:
            fig, ax = plt.subplots(1,4, figsize=(24, 6), sharey=True)
            # at each depth, plot the stacked fraction of each phase, where depth is the y-axis
            bottom = np.zeros(len(self.fraction))
            
            for i ,col in enumerate(self.phase_cols):
                ax[0].barh(self.fraction['depth']+interval, self.fraction[col], left=bottom, height=interval*2, label=col, color=self.color_dict[col])
                bottom += self.fraction[col]
            ax[0].set_xlabel('Volume Fraction')
            ax[0].set_ylabel('Depth (km)')
            #ax[0].set_ylim(2800, 25)
            ax[0].set_ylim(self.fraction['depth'].max(), self.fraction['depth'].min())
            #ax[0].invert_yaxis()  # invert y-axis to have depth increase downwards
            ax[0].minorticks_on()
            ax[0].set_xlim(0, 1)
            ax[0].tick_params(axis='both', direction='in', length=3, width=1, which='both')
            


            
            
            depth, P = self.fraction['depth'].to_numpy(), self.fraction['P'].to_numpy()
            if depth[0] > depth[-1]:                # make depth ascend
                    depth, P = depth[::-1], P[::-1]
            ax2 = ax[0].twinx()                 # right-hand y-axis
            ax2.set_ylim(ax[0].get_ylim())      # share depth ruler

            # --- major ticks: exactly where the primary ticks are -------------
            d_ticks = ax[0].get_yticks()        # depth positions already on left
            p_ticks = np.interp(d_ticks, depth, P)
            ax2.set_yticks(d_ticks)
            ax2.set_yticklabels([f'{p:.0f}' for p in p_ticks])

            # --- minor ticks: evenly spaced, no labels ------------------------
            ax2.yaxis.set_minor_locator(AutoMinorLocator())        # default = 4 minor gaps

            # --- tick appearance ---------------------------------------------
            ax2.tick_params(axis='y', which='major', length=6, direction='in')
            ax2.tick_params(axis='y', which='minor', length=3, direction='in')  # shorter
            ax2.set_ylabel('Pressure (GPa)')
            if pspl_boundary:
                def pspl_ppv_boundary(T):
                    """
                    Calculate the pspl-ppv boundary temperature.
                    """
                    return 450 - 15e6*T/1e9
                x_position = np.linspace(0, 1, len(self.fraction['T']))
                y_position = pspl_ppv_boundary(self.fraction['T'])
                ax2.plot(x_position, y_position, color='k', linestyle='--', label='pspl-ppv boundary')
            
            ax_twiny = ax[0].twiny()
            ax_twiny.plot(self.fraction['T'], self.fraction['depth'], color='k', linestyle=':')
            ax_twiny.set_xlabel(r'T (K)',color='k')
            ax_twiny.tick_params(axis='x', direction='in', length=5, width=1, colors='k', which='both')
            #ax_twiny.set_xlim(1750, 2600)
            
            
            # twin x-axis is VB, plot VB versus depth using the profile data
            ax[1].plot(self.profile['VP'], self.profile['depth'] + interval, color='r', linestyle=':')
            ax[1].set_xlabel(r'V$_P$ (km/s)')
            ax[1].set_ylabel('Depth (km)')
            ax[1].set_xlim(5, 15)
            
            # twin x-axis is VS, plot VS versus depth using the profile data
            ax[2].plot(self.profile['VS'], self.profile['depth'] + interval, color='r', linestyle=':')
            ax[2].set_xlabel(r'V$_S$ (km/s)')
            ax[2].set_ylabel('Depth (km)')
            ax[2].set_xlim(3, 8)
            
            # twin x-axis is VS, plot VS versus depth using the profile data
            ax[3].plot(self.profile['rho'], self.profile['depth'] + interval, color='r', linestyle=':')
            ax[3].set_xlabel(r'$\rho$ (g/cm$^3$)')
            ax[3].set_ylabel('Depth (km)')
            ax[3].set_xlim(3, 6)
            
            plt.title(self.title, fontsize=20)
            
            if contrast:
                # read PREM model
                prem_dir = './Real_run_test/REM1D_layer_properties.txt'
                prem = pd.read_csv(prem_dir, delim_whitespace=True, comment='#', skiprows=22)
                # Headers: Layer Radius  Depth       Rho      Vph        Vpv       Vp       Vsh      Vsv      Vs      ETA      A    C     N     L     F      QMU   QKAPPA    MU  KAPPA POISSON Pressure   dKdP   Bullen   Gravity
                # Provide a proper mapping for renaming columns. Adjust new names as needed.
                rename_dict = {
                    'Layer': 'Layer',
                    'Radius': 'Radius',
                    'Depth': 'Depth',
                    'Rho': 'Rho',
                    'Vph': 'Vph',
                    'Vpv': 'Vpv',
                    'Vp': 'Vp',
                    'Vsh': 'Vsh',
                    'Vsv': 'Vsv',
                    'Vs': 'Vs',
                    'ETA': 'ETA',
                    'A': 'A',
                    'C': 'C',
                    'N': 'N',
                    'L': 'L',
                    'F': 'F',
                    'QMU': 'QMU',
                    'QKAPPA': 'QKAPPA',
                    'MU': 'MU',
                    'KAPPA': 'KAPPA',
                    'POISSON': 'POISSON',
                    'Pressure': 'Pressure',
                    'dKdP': 'dKdP',
                    'Bullen': 'Bullen',
                    'Gravity': 'Gravity'
                }
                prem.rename(columns=rename_dict, inplace=True)

            
                ax[1].plot(prem['Vp'].values[370:-20], prem['Depth'].values[370:-20], color='black', linestyle='--')
                ax[2].plot(prem['Vs'].values[370:-20], prem['Depth'].values[370:-20], color='black', linestyle='--')
                ax[3].plot(prem['Rho'].values[370:-20], prem['Depth'].values[370:-20], color='black', linestyle='--')
            
            
                prem_orgin_dir = './Real_run_test/PREM_1s_IDV.csv'
                prem_orgin = pd.read_csv(prem_orgin_dir, comment='#', skiprows=1)
                # Headers: radius[unit="km"],depth[unit="km"],density[unit="g/cm^3"],Vpv[unit="km/s"],Vph[unit="km/s"],Vsv[unit="km/s"],Vsh[unit="km/s"],eta[unit=""],Q-mu[unit=""],Q-kappa[unit=""]
                # Provide a proper mapping for renaming columns. Adjust new names as needed.
                rename_dict = {
                    'radius[unit="km"]': 'radius',
                    'depth[unit="km"]': 'depth',
                    'density[unit="g/cm^3"]': 'density',
                    'Vpv[unit="km/s"]': 'Vpv',
                    'Vph[unit="km/s"]': 'Vph',
                    'Vsv[unit="km/s"]': 'Vsv',
                    'Vsh[unit="km/s"]': 'Vsh',
                    'eta[unit=""]': 'eta',
                    'Q-mu[unit=""]': 'Q-mu',
                    'Q-kappa[unit=""]': 'Q-kappa'
                }
                prem_orgin.rename(columns=rename_dict, inplace=True)
                # create two new columns for Vp and Vs
                prem_orgin['Vp'] = np.sqrt((prem_orgin['Vpv']**2 + prem_orgin['Vph']**2)/2)
                prem_orgin['Vs'] = np.sqrt((prem_orgin['Vsv']**2 + prem_orgin['Vsh']**2)/2)
                
                # plot the Vp and Vs
                ax[1].plot(prem_orgin['Vp'].values[0:168], prem_orgin['depth'].values[0:168], color='grey', linestyle='-')
                
                ax[2].plot(prem_orgin['Vs'].values[0:168], prem_orgin['depth'].values[0:168], color='grey', linestyle='-')
                ax[3].plot(prem_orgin['density'].values[0:168], prem_orgin['depth'].values[0:168], color='grey', linestyle='-')
                
            plt.tight_layout()
            ax[0].legend(bbox_to_anchor=(-0.8, 1), loc='upper left')
        else:
            fig, ax = plt.subplots(1,1, figsize=(6, 6), sharey=True)
            plt.title(self.title, fontsize=20)
            if ticks_type == 'depth':
                # at each depth, plot the stacked fraction of each phase, where depth is the y-axis
                bottom = np.zeros(len(self.fraction))
                
                for i ,col in enumerate(self.phase_cols):
                    ax.barh(self.fraction['depth']+interval, self.fraction[col], left=bottom, height=interval*2, label=col, color=self.color_dict[col])
                    bottom += self.fraction[col]
                ax.set_xlabel('Volume Fraction')
                ax.set_ylabel('Depth (km)')
                #ax.set_ylim(2800, 25)
                ax.set_ylim(self.fraction['depth'].max(), self.fraction['depth'].min())
                #ax.invert_yaxis()  # invert y-axis to have depth increase downwards
                ax.minorticks_on()
                ax.set_xlim(0, 1)
                ax.tick_params(axis='both', direction='in', length=3, width=1, which='both')
                
                depth, P = self.fraction['depth'].to_numpy(), self.fraction['P'].to_numpy()
                if depth[0] > depth[-1]:                # make depth ascend
                        depth, P = depth[::-1], P[::-1]
                ax2 = ax.twinx()                 # right-hand y-axis
                ax2.set_ylim(ax.get_ylim())      # share depth ruler

                # --- major ticks: exactly where the primary ticks are -------------
                d_ticks = ax.get_yticks()        # depth positions already on left
                p_ticks = np.interp(d_ticks, depth, P)
                ax2.set_yticks(d_ticks)
                ax2.set_yticklabels([f'{p:.0f}' for p in p_ticks])

                # --- minor ticks: evenly spaced, no labels ------------------------
                ax2.yaxis.set_minor_locator(AutoMinorLocator())        # default = 4 minor gaps

                # --- tick appearance ---------------------------------------------
                ax2.tick_params(axis='y', which='major', length=6, direction='in')
                ax2.tick_params(axis='y', which='minor', length=3, direction='in')  # shorter
                ax2.set_ylabel('Pressure (GPa)')
                
                ax_twiny = ax.twiny()
                ax_twiny.plot(self.fraction['T'], self.fraction['depth'], color='k', linestyle=':')
                ax_twiny.set_xlabel(r'T (K)',color='k')
                ax_twiny.tick_params(axis='x', direction='in', length=5, width=1, colors='k', which='both')
                ax.legend(bbox_to_anchor=(-0.8, 1), loc='upper left')
                #ax_twiny.set_xlim(1750, 2600)
            else:
                # at each depth, plot the stacked fraction of each phase, where depth is the y-axis
                interval = self.fraction['P'].max()/len(self.fraction['P'])
                bottom = np.zeros(len(self.fraction))
                
                for i ,col in enumerate(self.phase_cols):
                    ax.barh(self.fraction['P']+interval, self.fraction[col], left=bottom, height=interval*2, label=col, color=self.color_dict[col])
                    bottom += self.fraction[col]
                ax.set_xlabel('Volume Fraction')
                ax.set_ylabel('P (GPa)')
                #ax.set_ylim(2800, 25)
                ax.set_ylim(self.fraction['P'].max(), self.fraction['P'].min())
                #ax.invert_yaxis()  # invert y-axis to have depth increase downwards
                ax.minorticks_on()
                ax.set_xlim(0, 1)
                ax.tick_params(axis='both', direction='in', length=3, width=1, which='both')
                if pspl_boundary:
                    def pspl_ppv_boundary(T):
                        """
                        Calculate the pspl-ppv boundary temperature.
                        """
                        return 450 - 15e6*T/1e9
                    x_position = np.linspace(0, 1, len(self.fraction['T']))
                    y_position = pspl_ppv_boundary(self.fraction['T'])
                    ax.plot(x_position, y_position, color='r', linestyle='-', label='pspl-ppv boundary')
                    print('y position:', y_position[0])
                    #print('x position:', x_position)
                
                ax_twiny = ax.twiny()
                ax_twiny.plot(self.fraction['T'], self.fraction['P'], color='k', linestyle=':')
                ax_twiny.set_xlabel(r'T (K)',color='k')
                ax_twiny.tick_params(axis='x', direction='in', length=5, width=1, colors='k', which='both')
                ax.legend(bbox_to_anchor=(-0.8, 1), loc='upper left')
                #ax_twiny.set_xlim(1750, 2600)
                

                
    def density_plot(self, ticks_type='pressure',save = False):
        """
        Plot the density vs depth for a specific mineral.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        if ticks_type == 'depth':
            ax.plot(self.fraction['depth'], self.profile['rho'], color='k', linestyle='-')
            ax.set_xlabel('Depth (km)')
            ax.set_ylabel(r'$\rho$ (g/cm$^3$)')
            ax.set_ylim(self.fraction['depth'].max(), self.fraction['depth'].min())
            ax.invert_yaxis()
            ax.minorticks_on()
            ax.set_xlim(0, 6)
        else:
            ax.plot(self.profile['rho'], self.fraction['P'], color='k', linestyle='-')
            ax.set_xlabel(r'$\rho$ (g/cm$^3$)')
            ax.set_ylabel('P (GPa)')
            ax.set_ylim(self.fraction['P'].max(), self.fraction['P'].min())
            ax.invert_yaxis()
            ax.minorticks_on()
            if save:
                P_rho_profile = pd.DataFrame({'P': self.fraction['P'], 'rho': self.profile['rho']})
                P_rho_profile.to_csv('/projects/JIEDENG/Donghao/Group_Tutorial/HeFESTo_modified/For_Maggie/Water_storage/data/P_rho_profile.csv', index=False)

                  
                
    def find_depth(self, mute=False,phase_name='ppv',boundary='base'):
        """
        Find the depth of the last occurrence of ppv phase. If not found, find the first occurrence of pspl phase.
        """
        if boundary not in ['top', 'base']:
            raise ValueError("boundary must be either 'top' or 'base'")
        
        if phase_name not in self.phase_cols:
            raise ValueError(f"Phase '{phase_name}' not found in the fraction data. Available phases: {self.phase_cols}")
        
        if boundary == 'top':
            phase_depth = self.fraction[self.fraction[phase_name] > 0]['depth'].min()
        elif boundary == 'base':
            phase_depth = self.fraction[self.fraction[phase_name] > 0]['depth'].max()
        
        if mute == False:
            if np.isnan(phase_depth):
                print(f"No {phase_name} phase found.")
                return np.nan
            else:
                print(f"{phase_name} phase {boundary} found at depth: {phase_depth} km")
        return phase_depth
    
    def find_pressure(self, mute=False, phase_name='ppv', boundary='base'):
        """
        Find the pressure of the last occurrence of ppv phase. If not found, find the first occurrence of pspl phase.
        """
        if boundary not in ['top', 'base']:
            raise ValueError("boundary must be either 'top' or 'base'")
        
        if phase_name not in self.phase_cols:
            return -1
            #raise ValueError(f"Phase '{phase_name}' not found in the fraction data. Available phases: {self.phase_cols}")
        
        if boundary == 'top':
            phase_pressure = self.fraction[self.fraction[phase_name] > 0]['P'].min()
        elif boundary == 'base':
            phase_pressure = self.fraction[self.fraction[phase_name] > 0]['P'].max()
        
        if mute == False:
            if np.isnan(phase_pressure):
                print(f"No {phase_name} phase found.")
                return np.nan
            else:
                print(f"{phase_name} phase {boundary} found at pressure: {phase_pressure} GPa")
        return phase_pressure
    
class Multi_Simulations:
    def __init__(self, sim_list):
        self.sim_list = sim_list
        self.simulations = []
        for sim in self.sim_list:
            self.simulations.append(Simulation(sim['name'], sim['title'], sim['dir']))
            
    def Mineral_plot(self):
        """
        Plot the mineral fraction vs depth for a specific mineral.
        """
        fig, ax = plt.subplots(1, len(self.sim_list), figsize=(6*len(self.sim_list), 6))
        # at each depth, plot the stacked fraction of each phase, where depth is the y-axis

        for i, sim in enumerate(self.simulations):
            sim.read_phase_fraction(save=False, fraction_type='volume')
            maximum_Depth = sim.fraction['depth'].max()
            interval = maximum_Depth/len(sim.fraction['P'])
            bottom = np.zeros(len(sim.fraction))
            for j ,col in enumerate(sim.phase_cols):
                ax[i].barh(sim.fraction['depth']+interval, sim.fraction[col], left=bottom, height=interval*2, label=col, color=sim.color_dict[col])
                bottom += sim.fraction[col]
            
            ax[i].set_xlabel('Volume Fraction')
            ax[i].set_ylabel('Depth (km)')
            ax[i].set_ylim(2600, 96)
            ax[i].minorticks_on()
            ax[i].set_xlim(0, 1)
            ax[i].tick_params(axis='both', direction='in', length=6, width=1, which='major')
            ax[i].tick_params(axis='both', direction='in', length=3, width=1, which='minor')
            ax[i].set_title(sim.title)
            ax[i].xaxis.set_tick_params(pad=10)
            
            depth, P = sim.fraction['depth'].to_numpy(), sim.fraction['P'].to_numpy()
            if depth[0] > depth[-1]:                # make depth ascend
                depth, P = depth[::-1], P[::-1]
            ax2 = ax[i].twinx()                 # right-hand y-axis
            ax2.set_ylim(ax[i].get_ylim())      # share depth ruler

            # --- major ticks: exactly where the primary ticks are -------------
            d_ticks = ax[i].get_yticks()        # depth positions already on left
            p_ticks = np.interp(d_ticks, depth, P)
            ax2.set_yticks(d_ticks)
            ax2.set_yticklabels([f'{p:.0f}' for p in p_ticks])

            # --- minor ticks: evenly spaced, no labels ------------------------
            ax2.yaxis.set_minor_locator(AutoMinorLocator())        # default = 4 minor gaps

            # --- tick appearance ---------------------------------------------
            ax2.tick_params(axis='y', which='major', length=6, direction='in')
            ax2.tick_params(axis='y', which='minor', length=3, direction='in')  # shorter
            ax2.set_ylabel('Pressure (GPa)')
            
            
            ax_twiny = ax[i].twiny()
            ax_twiny.plot(sim.fraction['T'], sim.fraction['depth'], color='r', linestyle='-',linewidth=2)
            ax_twiny.set_xlabel(r'T (K)',color='k')
            ax_twiny.minorticks_on()
            ax_twiny.tick_params(axis='x', direction='in', length=6, width=1, colors='k', which='major')
            ax_twiny.tick_params(axis='x', direction='in', length=3, width=1, colors='k', which='minor')
            ax_twiny.yaxis.set_tick_params(pad=10)
            #ax_twiny.set_xlim(1620, 2580)
            
            
            # put the capital letter in the top right corner
            ax[i].text(0.15, 0.1, f'({chr(65+i)})', transform=ax[i].transAxes, fontsize=20,
                    verticalalignment='top', horizontalalignment='right', color='black', weight='bold')
        
        plt.tight_layout()
        ax[0].legend(bbox_to_anchor=(-1.2, 1), loc='upper left',ncol=2, fancybox=False, frameon=True, edgecolor='black')
       
        
    def Properties_plot(self,reference='PREM'):
        """
        Plot the mineral fraction vs depth for a specific mineral.
        """
        fig, ax = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        
        for i, sim in enumerate(self.simulations):
            sim.read_profile(save=False)
            if reference != 'PREM':
                prem_dir = './Real_run_test/REM1D_layer_properties.txt'
                prem = pd.read_csv(prem_dir, delim_whitespace=True, comment='#', skiprows=22)
                # Headers: Layer Radius  Depth       Rho      Vph        Vpv       Vp       Vsh      Vsv      Vs      ETA      A    C     N     L     F      QMU   QKAPPA    MU  KAPPA POISSON Pressure   dKdP   Bullen   Gravity
                # Provide a proper mapping for renaming columns. Adjust new names as needed.
                rename_dict = {
                    'Layer': 'Layer',
                    'Radius': 'Radius',
                    'Depth': 'Depth',
                    'Rho': 'Rho',
                    'Vph': 'Vph',
                    'Vpv': 'Vpv',
                    'Vp': 'Vp',
                    'Vsh': 'Vsh',
                    'Vsv': 'Vsv',
                    'Vs': 'Vs',
                    'ETA': 'ETA',
                    'A': 'A',
                    'C': 'C',
                    'N': 'N',
                    'L': 'L',
                    'F': 'F',
                    'QMU': 'QMU',
                    'QKAPPA': 'QKAPPA',
                    'MU': 'MU',
                    'KAPPA': 'KAPPA',
                    'POISSON': 'POISSON',
                    'Pressure': 'Pressure',
                    'dKdP': 'dKdP',
                    'Bullen': 'Bullen',
                    'Gravity': 'Gravity'
                }
                prem.rename(columns=rename_dict, inplace=True)
        
                numeric_cols = ['Depth', 'Vp', 'Vs', 'Rho']
                prem[numeric_cols] = prem[numeric_cols].apply(pd.to_numeric, errors='coerce')
                prem = (prem.dropna(subset=numeric_cols)      # get rid of blank lines, if any
                        .sort_values('Depth')             # *ascending* (0 km at surface → 6371 km at CMB)
                        )
                
                depth_grid = prem['Depth'].to_numpy()   # km
                vp_grid    = prem['Vp'].to_numpy()
                vs_grid    = prem['Vs'].to_numpy()
                rho_grid   = prem['Rho'].to_numpy()
                
                # Interpolate the Vp and Vs to the same depth as the profile data
                new_depth = sim.profile['depth'].values
                interp_Vp  = np.interp(new_depth, depth_grid, vp_grid,  left=np.nan, right=np.nan)
                interp_Vs  = np.interp(new_depth, depth_grid, vs_grid,  left=np.nan, right=np.nan)
                interp_Rho = np.interp(new_depth, depth_grid, rho_grid, left=np.nan, right=np.nan)
    
            else:
                prem_orgin_dir = './Real_run_test/PREM_1s_IDV.csv'
                prem_orgin = pd.read_csv(prem_orgin_dir, comment='#', skiprows=1)
                # Headers: radius[unit="km"],depth[unit="km"],density[unit="g/cm^3"],Vpv[unit="km/s"],Vph[unit="km/s"],Vsv[unit="km/s"],Vsh[unit="km/s"],eta[unit=""],Q-mu[unit=""],Q-kappa[unit=""]
                # Provide a proper mapping for renaming columns. Adjust new names as needed.
                rename_dict = {
                    'radius[unit="km"]': 'radius',
                    'depth[unit="km"]': 'depth',
                    'density[unit="g/cm^3"]': 'density',
                    'Vpv[unit="km/s"]': 'Vpv',
                    'Vph[unit="km/s"]': 'Vph',
                    'Vsv[unit="km/s"]': 'Vsv',
                    'Vsh[unit="km/s"]': 'Vsh',
                    'eta[unit=""]': 'eta',
                    'Q-mu[unit=""]': 'Q-mu',
                    'Q-kappa[unit=""]': 'Q-kappa'
                }
                prem_orgin.rename(columns=rename_dict, inplace=True)
                # create two new columns for Vp and Vs
                prem_orgin['Vp'] = np.sqrt((prem_orgin['Vpv']**2 + prem_orgin['Vph']**2)/2)
                prem_orgin['Vs'] = np.sqrt((prem_orgin['Vsv']**2 + prem_orgin['Vsh']**2)/2)
                
                
                # Interpolate the density to the same depth as the profile data
                numeric_cols = ['depth', 'Vp', 'Vs', 'density']
                prem_orgin[numeric_cols] = prem_orgin[numeric_cols].apply(pd.to_numeric, errors='coerce')
                prem_orgin = (prem_orgin.dropna(subset=numeric_cols)      # get rid of blank lines, if any
                        .sort_values('depth')             # *ascending* (0 km at surface → 6371 km at CMB)
                        )
                
                depth_grid = prem_orgin['depth'].to_numpy()   # km
                vp_grid    = prem_orgin['Vp'].to_numpy()
                vs_grid    = prem_orgin['Vs'].to_numpy()
                rho_grid   = prem_orgin['density'].to_numpy()

                # Interpolate the Vp and Vs to the same depth as the profile data
                new_depth = sim.profile['depth'].values
                interp_Vp  = np.interp(new_depth, depth_grid, vp_grid,  left=np.nan, right=np.nan)
                interp_Vs  = np.interp(new_depth, depth_grid, vs_grid,  left=np.nan, right=np.nan)
                interp_Rho = np.interp(new_depth, depth_grid, rho_grid, left=np.nan, right=np.nan)
          
            maximum_Depth = sim.profile['depth'].max()
            interval = maximum_Depth/len(sim.profile['depth'])
            bottom = np.zeros(len(sim.profile))
            ax[0].plot((sim.profile['VP']-interp_Vp)/interp_Vp*100, sim.profile['depth'], color='C'+str(i), linestyle='-', label=sim.title)
            ax[0].vlines(x=0, ymin=0,ymax=3000,color='black', linestyle='--')
            ax[0].set_xlabel(r'dV$_P$ (%)')
            ax[0].set_ylabel('Depth (km)')
            ax[0].set_ylim(2600, 96)
            ax[0].minorticks_on()
            ax[0].tick_params(axis='both', direction='in', length=6, width=1, which='major')
            ax[0].tick_params(axis='both', direction='in', length=3, width=1, which='minor')
            
            
            ax[1].plot((sim.profile['VS']-interp_Vs)/interp_Vs*100, sim.profile['depth'], color='C'+str(i), linestyle='-', label=sim.title)
            ax[1].vlines(x=0, ymin=0,ymax=3000, color='black', linestyle='--')
            ax[1].set_xlabel(r'dV$_S$ (%)')
            ax[1].set_ylabel('Depth (km)')
            ax[1].set_ylim(2600,96)
            ax[1].minorticks_on()
            ax[1].tick_params(axis='both', direction='in', length=6, width=1, which='major')
            ax[1].tick_params(axis='both', direction='in', length=3, width=1, which='minor')
            
            
            ax[2].plot((sim.profile['rho']-interp_Rho)/interp_Rho*100, sim.profile['depth'], color='C'+str(i), linestyle='-', label=sim.title)
            ax[2].vlines(x=0,ymin=0,ymax=3000, color='black', linestyle='--')
            ax[2].set_xlabel(r'd$\rho$ (%)')
            ax[2].set_ylabel('Depth (km)')
            ax[2].set_ylim(2600, 96)
            ax[2].minorticks_on()
            ax[2].tick_params(axis='both', direction='in', length=6, width=1, which='major')
            ax[2].tick_params(axis='both', direction='in', length=3, width=1, which='minor')
            
        
        depth, P = sim.fraction['depth'].to_numpy(), sim.fraction['P'].to_numpy()
        if depth[0] > depth[-1]:                # make depth ascend
            depth, P = depth[::-1], P[::-1]
        for i in range(3):
            ax2 = ax[i].twinx()                 # right-hand y-axis
            ax2.set_ylim(ax[i].get_ylim())      # share depth ruler

            # --- major ticks: exactly where the primary ticks are -------------
            d_ticks = ax[i].get_yticks()        # depth positions already on left
            p_ticks = np.interp(d_ticks, depth, P)
            ax2.set_yticks(d_ticks)
            ax2.set_yticklabels([f'{p:.0f}' for p in p_ticks])

            # --- minor ticks: evenly spaced, no labels ------------------------
            ax2.yaxis.set_minor_locator(AutoMinorLocator())        # default = 4 minor gaps

            # --- tick appearance ---------------------------------------------
            ax2.tick_params(axis='y', which='major', length=6, direction='in')
            ax2.tick_params(axis='y', which='minor', length=3, direction='in')  # shorter
            ax2.set_ylabel('Pressure (GPa)')
            # panel label (optional) ------------------------------------------
            ax[i].text(0.15, 0.1, f'({chr(65+i)})',
                    transform=ax[i].transAxes,
                    fontsize=20, fontweight='bold',
                    verticalalignment='top', horizontalalignment='right')

         
        
        ax[0].legend(bbox_to_anchor=(-0.6, .8), fancybox=False, frameon=True, edgecolor='black')
        plt.tight_layout()
        
 