"""
Process Engineering Library

A comprehensive library for process engineering calculations and utilities,
specifically designed for refinery operations and petroleum processing.

Author: Created for refinery process engineering applications
Date: June 11, 2025
"""

import math
import numpy as np
from typing import Union, List, Tuple, Optional


def welcome_message():
    """
    Display welcome message for the process engineering library.
    """
    print("Welcome to the Process Engineering Library!")
    print("This library contains functions for refinery and petroleum processing calculations.")
    print("Ready to add your custom functions...")


# ============================================================================
# UNIT CONVERSION SYSTEM
# ============================================================================

class UnitConverter:
    """
    Comprehensive unit conversion system for process engineering calculations.
    All internal calculations use: barg, Celsius, kg/m³, kg/kmol
    """
    
    @staticmethod
    def pressure_to_barg(value: float, from_unit: str) -> float:
        """
        Convert pressure to barg (gauge pressure in bar).
        
        Args:
            value (float): Pressure value
            from_unit (str): Source unit ("barg", "bara", "psig", "psia", "kpa", "mpa", "atm")
            
        Returns:
            float: Pressure in barg
        """
        from_unit = from_unit.lower()
        
        # First convert to bara (absolute bar)
        if from_unit == "barg":
            bara = value + 1.01325  # Add atmospheric pressure
        elif from_unit == "bara":
            bara = value
        elif from_unit == "psig":
            bara = (value + 14.696) * 0.0689476  # Convert psia to bara
        elif from_unit == "psia":
            bara = value * 0.0689476
        elif from_unit == "kpa":
            bara = value * 0.01
        elif from_unit == "mpa":
            bara = value * 10
        elif from_unit == "atm":
            bara = value * 1.01325
        else:
            raise ValueError(f"Unsupported pressure unit: {from_unit}")
        
        # Convert bara to barg
        return bara - 1.01325
    
    @staticmethod
    def temperature_to_celsius(value: float, from_unit: str) -> float:
        """
        Convert temperature to Celsius.
        
        Args:
            value (float): Temperature value
            from_unit (str): Source unit ("c", "celsius", "f", "fahrenheit", "k", "kelvin", "r", "rankine")
            
        Returns:
            float: Temperature in Celsius
        """
        from_unit = from_unit.lower()
        
        if from_unit in ["c", "celsius"]:
            return value
        elif from_unit in ["f", "fahrenheit"]:
            return (value - 32) * 5/9
        elif from_unit in ["k", "kelvin"]:
            return value - 273.15
        elif from_unit in ["r", "rankine"]:
            return (value - 491.67) * 5/9
        else:
            raise ValueError(f"Unsupported temperature unit: {from_unit}")
    
    @staticmethod
    def density_to_kg_m3(value: float, from_unit: str) -> float:
        """
        Convert density to kg/m³.
        
        Args:
            value (float): Density value
            from_unit (str): Source unit ("kg/m3", "g/cm3", "lb/ft3", "lb/gal", "api")
            
        Returns:
            float: Density in kg/m³
        """
        from_unit = from_unit.lower().replace("/", "_").replace("³", "3")
        
        if from_unit in ["kg_m3", "kg/m3"]:
            return value
        elif from_unit in ["g_cm3", "g/cm3"]:
            return value * 1000
        elif from_unit in ["lb_ft3", "lb/ft3"]:
            return value * 16.0185
        elif from_unit in ["lb_gal", "lb/gal"]:
            return value * 119.826
        elif from_unit == "api":
            # API gravity to kg/m³: ρ = 141.5/(131.5 + API) * 1000
            return 141.5 / (131.5 + value) * 1000
        else:
            raise ValueError(f"Unsupported density unit: {from_unit}")
    
    @staticmethod
    def molecular_weight_to_kg_kmol(value: float, from_unit: str) -> float:
        """
        Convert molecular weight to kg/kmol.
        
        Args:
            value (float): Molecular weight value
            from_unit (str): Source unit ("kg/kmol", "g/mol", "lb/lbmol")
            
        Returns:
            float: Molecular weight in kg/kmol
        """
        from_unit = from_unit.lower().replace("/", "_")
        
        if from_unit in ["kg_kmol", "kg/kmol"]:
            return value
        elif from_unit in ["g_mol", "g/mol"]:
            return value  # g/mol = kg/kmol
        elif from_unit in ["lb_lbmol", "lb/lbmol"]:
            return value * 0.453592  # lb to kg
        else:
            raise ValueError(f"Unsupported molecular weight unit: {from_unit}")
    
    @staticmethod
    def viscosity_to_cp(value: float, from_unit: str) -> float:
        """
        Convert viscosity to centiPoise (cP).
        
        Args:
            value (float): Viscosity value
            from_unit (str): Source unit ("cp", "pas", "poise", "ssu", "cst")
            
        Returns:
            float: Viscosity in cP
        """
        from_unit = from_unit.lower()
        
        if from_unit == "cp":
            return value
        elif from_unit in ["pas", "pa.s"]:
            return value * 1000
        elif from_unit == "poise":
            return value * 100
        elif from_unit == "ssu":  # Saybolt Universal Seconds (approximate)
            return 4.632 * value - 39.6 if value > 100 else 4.632 * value
        elif from_unit == "cst":  # Approximate conversion (needs density)
            return value  # Simplified - actual conversion needs density
        else:
            raise ValueError(f"Unsupported viscosity unit: {from_unit}")


# ============================================================================
# ENHANCED FLOWMETER CORRECTION CLASSES
# ============================================================================

class FlowmeterBase:
    """
    Enhanced base class for all flowmeter types with automatic unit conversion.
    All internal calculations use: barg, Celsius, kg/m³, kg/kmol
    """
    
    def __init__(self, 
                 design_pressure: float,
                 design_temperature: float,
                 actual_pressure: float,
                 actual_temperature: float,
                 fluid_type: str = "liquid",
                 pressure_unit: str = "barg",
                 temperature_unit: str = "celsius"):
        """
        Initialize base flowmeter parameters with automatic unit conversion.
        
        Args:
            design_pressure (float): Design pressure
            design_temperature (float): Design temperature
            actual_pressure (float): Actual operating pressure
            actual_temperature (float): Actual operating temperature
            fluid_type (str): Type of fluid ("liquid" or "gas")
            pressure_unit (str): Pressure unit ("barg", "bara", "psig", "psia", "kpa", "mpa", "atm")
            temperature_unit (str): Temperature unit ("c", "celsius", "f", "fahrenheit", "k", "kelvin")
        """
        # Convert to standard units
        self.design_pressure = UnitConverter.pressure_to_barg(design_pressure, pressure_unit)
        self.design_temperature = UnitConverter.temperature_to_celsius(design_temperature, temperature_unit)
        self.actual_pressure = UnitConverter.pressure_to_barg(actual_pressure, pressure_unit)
        self.actual_temperature = UnitConverter.temperature_to_celsius(actual_temperature, temperature_unit)
        
        self.fluid_type = fluid_type.lower()
        
        # Store original input units for display
        self.input_pressure_unit = pressure_unit
        self.input_temperature_unit = temperature_unit
        
        # Initialize fluid properties (to be set by methods)
        self.design_density = None  # kg/m³
        self.actual_density = None  # kg/m³
        self.design_molecular_weight = None  # kg/kmol
        self.actual_molecular_weight = None  # kg/kmol
        self.design_viscosity = None  # cP
        self.actual_viscosity = None  # cP
        
        # Units for fluid properties
        self.density_unit = None
        self.molecular_weight_unit = None
        self.viscosity_unit = None
        
    def set_liquid_properties(self, 
                            design_density: float, 
                            actual_density: float,
                            design_viscosity: float = None, 
                            actual_viscosity: float = None,
                            density_unit: str = "kg/m3",
                            viscosity_unit: str = "cp"):
        """
        Set liquid-specific properties with automatic unit conversion.
        
        Args:
            design_density (float): Design density
            actual_density (float): Actual density
            design_viscosity (float): Design viscosity (optional)
            actual_viscosity (float): Actual viscosity (optional)
            density_unit (str): Density unit ("kg/m3", "g/cm3", "lb/ft3", "lb/gal", "api")
            viscosity_unit (str): Viscosity unit ("cp", "pas", "poise", "ssu", "cst")
        """
        self.design_density = UnitConverter.density_to_kg_m3(design_density, density_unit)
        self.actual_density = UnitConverter.density_to_kg_m3(actual_density, density_unit)
        
        if design_viscosity is not None:
            self.design_viscosity = UnitConverter.viscosity_to_cp(design_viscosity, viscosity_unit)
        if actual_viscosity is not None:
            self.actual_viscosity = UnitConverter.viscosity_to_cp(actual_viscosity, viscosity_unit)
            
        self.density_unit = density_unit
        self.viscosity_unit = viscosity_unit
        
    def set_gas_properties(self, 
                          design_molecular_weight: float, 
                          actual_molecular_weight: float,
                          design_density: float = None, 
                          actual_density: float = None,
                          molecular_weight_unit: str = "kg/kmol",
                          density_unit: str = "kg/m3"):
        """
        Set gas-specific properties with automatic unit conversion.
        
        Args:
            design_molecular_weight (float): Design molecular weight
            actual_molecular_weight (float): Actual molecular weight
            design_density (float): Design density (optional)
            actual_density (float): Actual density (optional)
            molecular_weight_unit (str): MW unit ("kg/kmol", "g/mol", "lb/lbmol")
            density_unit (str): Density unit ("kg/m3", "g/cm3", "lb/ft3")
        """
        self.design_molecular_weight = UnitConverter.molecular_weight_to_kg_kmol(
            design_molecular_weight, molecular_weight_unit)
        self.actual_molecular_weight = UnitConverter.molecular_weight_to_kg_kmol(
            actual_molecular_weight, molecular_weight_unit)
        
        if design_density is not None:
            self.design_density = UnitConverter.density_to_kg_m3(design_density, density_unit)
        if actual_density is not None:
            self.actual_density = UnitConverter.density_to_kg_m3(actual_density, density_unit)
            
        self.molecular_weight_unit = molecular_weight_unit
        self.density_unit = density_unit
            
    def calculate_correction_factor(self) -> float:
        """
        Calculate the flow correction factor.
        To be implemented by specific flowmeter subclasses.
        
        Returns:
            float: Correction factor to multiply with indicated flow
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def get_corrected_flow(self, indicated_flow: float) -> float:
        """
        Calculate corrected flow rate.
        
        Args:
            indicated_flow (float): Flow rate shown by the flowmeter
            
        Returns:
            float: Actual corrected flow rate
        """
        correction_factor = self.calculate_correction_factor()
        return indicated_flow * correction_factor
        
    def display_properties(self):
        """Display all flowmeter properties in standard units."""
        print("=" * 60)
        print(f"Flowmeter Type: {self.__class__.__name__}")
        print(f"Fluid Type: {self.fluid_type.upper()}")
        print("-" * 60)
        print("PROCESS CONDITIONS (Standard Units):")
        print(f"Design Pressure: {self.design_pressure:.3f} barg")
        print(f"Actual Pressure: {self.actual_pressure:.3f} barg")
        print(f"Design Temperature: {self.design_temperature:.2f} °C")
        print(f"Actual Temperature: {self.actual_temperature:.2f} °C")
        
        if self.fluid_type == "liquid":
            print("\nLIQUID PROPERTIES:")
            if self.design_density is not None:
                print(f"Design Density: {self.design_density:.2f} kg/m³")
            if self.actual_density is not None:
                print(f"Actual Density: {self.actual_density:.2f} kg/m³")
            if self.design_viscosity is not None:
                print(f"Design Viscosity: {self.design_viscosity:.3f} cP")
            if self.actual_viscosity is not None:
                print(f"Actual Viscosity: {self.actual_viscosity:.3f} cP")
                
        elif self.fluid_type == "gas":
            print("\nGAS PROPERTIES:")
            if self.design_molecular_weight is not None:
                print(f"Design Molecular Weight: {self.design_molecular_weight:.3f} kg/kmol")
            if self.actual_molecular_weight is not None:
                print(f"Actual Molecular Weight: {self.actual_molecular_weight:.3f} kg/kmol")
            if self.design_density is not None:
                print(f"Design Density: {self.design_density:.3f} kg/m³")
            if self.actual_density is not None:
                print(f"Actual Density: {self.actual_density:.3f} kg/m³")
        
        print("-" * 60)
        try:
            correction_factor = self.calculate_correction_factor()
            print(f"Correction Factor: {correction_factor:.6f}")
        except Exception as e:
            print(f"Correction Factor: Cannot calculate - {str(e)}")
        print("=" * 60)


class DPCellFlowmeter(FlowmeterBase):
    """
    Differential Pressure (DP) Cell Flowmeter.
    Includes orifice plates, venturi meters, and flow nozzles.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flowmeter_type = "DP Cell"
        
    def calculate_correction_factor(self) -> float:
        """
        Calculate correction factor for DP cell flowmeters.
        
        For DP flowmeters: Q_actual = Q_indicated × √(ρ_design/ρ_actual)
        
        Returns:
            float: Correction factor
        """
        if self.fluid_type == "liquid":
            if self.design_density is None or self.actual_density is None:
                raise ValueError("Density values required for liquid DP correction")
            return math.sqrt(self.design_density / self.actual_density)
            
        elif self.fluid_type == "gas":
            if (self.design_molecular_weight is None or self.actual_molecular_weight is None):
                raise ValueError("Molecular weight values required for gas DP correction")
            
            # For gases: ρ ∝ (P × MW) / T (using absolute temperature)
            # Convert barg to bara for gas density calculations
            design_pressure_bara = self.design_pressure + 1.01325
            actual_pressure_bara = self.actual_pressure + 1.01325
            
            rho_design_ratio = (design_pressure_bara * self.design_molecular_weight) / (self.design_temperature + 273.15)
            rho_actual_ratio = (actual_pressure_bara * self.actual_molecular_weight) / (self.actual_temperature + 273.15)
            
            return math.sqrt(rho_design_ratio / rho_actual_ratio)


class VenturiFlowmeter(DPCellFlowmeter):
    """
    Venturi Flowmeter - inherits from DP Cell as it follows same correction principles.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flowmeter_type = "Venturi"


class CoriolisFlowmeter(FlowmeterBase):
    """
    Coriolis Mass Flowmeter.
    Measures mass flow directly, minimal corrections needed.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flowmeter_type = "Coriolis"
        
    def calculate_correction_factor(self) -> float:
        """
        Coriolis flowmeters measure mass flow directly.
        Minimal correction needed, mostly for temperature effects on meter calibration.
        
        Returns:
            float: Correction factor (typically close to 1.0)
        """
        # Simplified correction for temperature effect on meter calibration
        # Typical temperature coefficient is around 0.01% per °C
        temp_coefficient = 0.0001  # per °C
        temp_difference = self.actual_temperature - self.design_temperature
        
        correction_factor = 1.0 + (temp_coefficient * temp_difference)
        return correction_factor


class VortexFlowmeter(FlowmeterBase):
    """
    Vortex Shedding Flowmeter.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flowmeter_type = "Vortex"
        
    def calculate_correction_factor(self) -> float:
        """
        Vortex flowmeters measure volumetric flow.
        Correction needed for density changes.
        
        Returns:
            float: Correction factor
        """
        if self.fluid_type == "liquid":
            if self.design_density is None or self.actual_density is None:
                raise ValueError("Density values required for liquid vortex correction")
            return self.design_density / self.actual_density
            
        elif self.fluid_type == "gas":
            if (self.design_molecular_weight is None or self.actual_molecular_weight is None):
                raise ValueError("Molecular weight values required for gas vortex correction")
            
            # For gases: ρ ∝ (P × MW) / T
            # Convert barg to bara for gas density calculations
            design_pressure_bara = self.design_pressure + 1.01325
            actual_pressure_bara = self.actual_pressure + 1.01325
            
            rho_design_ratio = (design_pressure_bara * self.design_molecular_weight) / (self.design_temperature + 273.15)
            rho_actual_ratio = (actual_pressure_bara * self.actual_molecular_weight) / (self.actual_temperature + 273.15)
            
            return rho_design_ratio / rho_actual_ratio


class UltrasonicFlowmeter(FlowmeterBase):
    """
    Ultrasonic Flowmeter (Transit-time or Doppler type).
    """
    
    def __init__(self, ultrasonic_type: str = "transit_time", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flowmeter_type = "Ultrasonic"
        self.ultrasonic_type = ultrasonic_type.lower()  # "transit_time" or "doppler"
        
    def calculate_correction_factor(self) -> float:
        """
        Ultrasonic flowmeters measure volumetric flow.
        Correction factors depend on type and fluid properties.
        
        Returns:
            float: Correction factor
        """
        if self.ultrasonic_type == "transit_time":
            # Transit-time: minimal correction for temperature effects on sound velocity
            # Sound velocity in liquids typically decreases with temperature
            if self.fluid_type == "liquid":
                # Approximate correction for sound velocity change
                temp_coefficient = -0.002  # per °C for typical liquids
                temp_difference = self.actual_temperature - self.design_temperature
                return 1.0 + (temp_coefficient * temp_difference)
            else:
                # For gases, sound velocity ∝ √T
                temp_design_k = self.design_temperature + 273.15
                temp_actual_k = self.actual_temperature + 273.15
                return math.sqrt(temp_design_k / temp_actual_k)
                
        elif self.ultrasonic_type == "doppler":
            # Doppler type: similar to other volumetric flowmeters
            return self._volumetric_correction()
            
    def _volumetric_correction(self) -> float:
        """Helper method for volumetric flow correction."""
        if self.fluid_type == "liquid":
            if self.design_density is None or self.actual_density is None:
                raise ValueError("Density values required for liquid ultrasonic correction")
            return self.design_density / self.actual_density
            
        elif self.fluid_type == "gas":
            if (self.design_molecular_weight is None or self.actual_molecular_weight is None):
                raise ValueError("Molecular weight values required for gas ultrasonic correction")
            
            # Convert barg to bara for gas density calculations
            design_pressure_bara = self.design_pressure + 1.01325
            actual_pressure_bara = self.actual_pressure + 1.01325
            
            rho_design_ratio = (design_pressure_bara * self.design_molecular_weight) / (self.design_temperature + 273.15)
            rho_actual_ratio = (actual_pressure_bara * self.actual_molecular_weight) / (self.actual_temperature + 273.15)
            
            return rho_design_ratio / rho_actual_ratio


class TurbineFlowmeter(FlowmeterBase):
    """
    Turbine Flowmeter.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flowmeter_type = "Turbine"
        
    def calculate_correction_factor(self) -> float:
        """
        Turbine flowmeters measure volumetric flow.
        Correction needed for viscosity and density effects.
        
        Returns:
            float: Correction factor
        """
        # Basic volumetric correction
        if self.fluid_type == "liquid":
            if self.design_density is None or self.actual_density is None:
                raise ValueError("Density values required for liquid turbine correction")
            
            base_correction = self.design_density / self.actual_density
            
            # Additional viscosity correction if available
            if self.design_viscosity and self.actual_viscosity:
                # Simplified viscosity correction (actual correction is more complex)
                viscosity_correction = 1.0 + 0.01 * math.log(self.actual_viscosity / self.design_viscosity)
                return base_correction * viscosity_correction
            
            return base_correction
            
        elif self.fluid_type == "gas":
            if (self.design_molecular_weight is None or self.actual_molecular_weight is None):
                raise ValueError("Molecular weight values required for gas turbine correction")
            
            # Convert barg to bara for gas density calculations
            design_pressure_bara = self.design_pressure + 1.01325
            actual_pressure_bara = self.actual_pressure + 1.01325
            
            rho_design_ratio = (design_pressure_bara * self.design_molecular_weight) / (self.design_temperature + 273.15)
            rho_actual_ratio = (actual_pressure_bara * self.actual_molecular_weight) / (self.actual_temperature + 273.15)
            
            return rho_design_ratio / rho_actual_ratio


# ============================================================================
# FLOWMETER UTILITY FUNCTIONS
# ============================================================================

def create_flowmeter(flowmeter_type: str, **kwargs) -> FlowmeterBase:
    """
    Factory function to create flowmeter objects.
    
    Args:
        flowmeter_type (str): Type of flowmeter ("dp_cell", "venturi", "coriolis", 
                             "vortex", "ultrasonic", "turbine")
        **kwargs: Flowmeter parameters
        
    Returns:
        FlowmeterBase: Appropriate flowmeter object
    """
    flowmeter_map = {
        "dp_cell": DPCellFlowmeter,
        "orifice": DPCellFlowmeter,
        "venturi": VenturiFlowmeter,
        "coriolis": CoriolisFlowmeter,
        "vortex": VortexFlowmeter,
        "ultrasonic": UltrasonicFlowmeter,
        "turbine": TurbineFlowmeter
    }
    
    flowmeter_type = flowmeter_type.lower()
    if flowmeter_type not in flowmeter_map:
        raise ValueError(f"Unknown flowmeter type: {flowmeter_type}")
        
    return flowmeter_map[flowmeter_type](**kwargs)


def batch_flowmeter_correction(flowmeters: List[FlowmeterBase], 
                              indicated_flows: List[float]) -> List[float]:
    """
    Apply corrections to multiple flowmeters at once.
    
    Args:
        flowmeters (List[FlowmeterBase]): List of flowmeter objects
        indicated_flows (List[float]): List of indicated flow rates
        
    Returns:
        List[float]: List of corrected flow rates
    """
    if len(flowmeters) != len(indicated_flows):
        raise ValueError("Number of flowmeters must match number of flow readings")
        
    corrected_flows = []
    for flowmeter, indicated_flow in zip(flowmeters, indicated_flows):
        corrected_flow = flowmeter.get_corrected_flow(indicated_flow)
        corrected_flows.append(corrected_flow)
        
    return corrected_flows


# ============================================================================
# DEMONSTRATION AND TESTING FUNCTIONS
# ============================================================================

def demo_flowmeter_corrections():
    """
    Demonstrate the flowmeter correction system with examples.
    """
    print("=" * 80)
    print("FLOWMETER CORRECTION SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: DP Cell Flowmeter for Liquid (with unit conversion)
    print("\n1. DP CELL FLOWMETER - LIQUID EXAMPLE")
    print("Input: Pressure in psig, Temperature in °F, Density in lb/ft³")
    
    dp_liquid = create_flowmeter(
        "dp_cell",
        design_pressure=150,      # psig
        design_temperature=70,    # °F
        actual_pressure=175,      # psig
        actual_temperature=95,    # °F
        fluid_type="liquid",
        pressure_unit="psig",
        temperature_unit="fahrenheit"
    )
    
    dp_liquid.set_liquid_properties(
        design_density=53.1,      # lb/ft³
        actual_density=51.8,      # lb/ft³
        density_unit="lb/ft3"
    )
    
    dp_liquid.display_properties()
    
    indicated_flow = 1000  # gpm
    corrected_flow = dp_liquid.get_corrected_flow(indicated_flow)
    print(f"Indicated Flow: {indicated_flow} gpm")
    print(f"Corrected Flow: {corrected_flow:.2f} gpm")
    
    # Example 2: Vortex Flowmeter for Gas (with unit conversion)
    print("\n\n2. VORTEX FLOWMETER - GAS EXAMPLE")
    print("Input: Pressure in bara, Temperature in K, MW in g/mol")
    
    vortex_gas = create_flowmeter(
        "vortex",
        design_pressure=5.0,      # bara
        design_temperature=298,   # K
        actual_pressure=4.2,      # bara
        actual_temperature=320,   # K
        fluid_type="gas",
        pressure_unit="bara",
        temperature_unit="kelvin"
    )
    
    vortex_gas.set_gas_properties(
        design_molecular_weight=28.97,  # g/mol (air)
        actual_molecular_weight=30.5,   # g/mol
        molecular_weight_unit="g/mol"
    )
    
    vortex_gas.display_properties()
    
    indicated_flow = 500  # Nm³/h
    corrected_flow = vortex_gas.get_corrected_flow(indicated_flow)
    print(f"Indicated Flow: {indicated_flow} Nm³/h")
    print(f"Corrected Flow: {corrected_flow:.2f} Nm³/h")
    
    # Example 3: Multiple Flowmeters (Batch Processing)
    print("\n\n3. BATCH PROCESSING EXAMPLE")
    
    flowmeters = [
        create_flowmeter("coriolis", 
                        design_pressure=10, design_temperature=25,
                        actual_pressure=12, actual_temperature=35),
        create_flowmeter("ultrasonic", 
                        design_pressure=8, design_temperature=20,
                        actual_pressure=9, actual_temperature=30,
                        ultrasonic_type="transit_time")
    ]
    
    indicated_flows = [100, 200]
    corrected_flows = batch_flowmeter_correction(flowmeters, indicated_flows)
    
    print("Batch Correction Results:")
    for i, (meter, indicated, corrected) in enumerate(zip(flowmeters, indicated_flows, corrected_flows)):
        print(f"Flowmeter {i+1} ({meter.__class__.__name__}): {indicated} → {corrected:.3f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    welcome_message()
    print("\nRunning flowmeter correction demonstration...")
    demo_flowmeter_corrections()
