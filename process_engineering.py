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
# FLOWMETER CORRECTION CLASSES
# ============================================================================

class FlowmeterBase:
    """
    Base class for all flowmeter types.
    Contains common properties and methods for flowmeter corrections.
    """
    
    def __init__(self, 
                 design_pressure: float,
                 design_temperature: float,
                 actual_pressure: float,
                 actual_temperature: float,
                 fluid_type: str = "liquid"):
        """
        Initialize base flowmeter parameters.
        
        Args:
            design_pressure (float): Design pressure (bar or psi)
            design_temperature (float): Design temperature (°C or °F)
            actual_pressure (float): Actual operating pressure (bar or psi)
            actual_temperature (float): Actual operating temperature (°C or °F)
            fluid_type (str): Type of fluid ("liquid" or "gas")
        """
        self.design_pressure = design_pressure
        self.design_temperature = design_temperature
        self.actual_pressure = actual_pressure
        self.actual_temperature = actual_temperature
        self.fluid_type = fluid_type.lower()
        
        # Initialize fluid properties (to be set by subclasses)
        self.design_density = None
        self.actual_density = None
        self.design_molecular_weight = None
        self.actual_molecular_weight = None
        self.design_viscosity = None
        self.actual_viscosity = None
        
    def set_liquid_properties(self, design_density: float, actual_density: float,
                            design_viscosity: float = None, actual_viscosity: float = None):
        """
        Set liquid-specific properties.
        
        Args:
            design_density (float): Design density (kg/m³ or lb/ft³)
            actual_density (float): Actual density (kg/m³ or lb/ft³)
            design_viscosity (float): Design viscosity (cP or Pa·s)
            actual_viscosity (float): Actual viscosity (cP or Pa·s)
        """
        self.design_density = design_density
        self.actual_density = actual_density
        self.design_viscosity = design_viscosity
        self.actual_viscosity = actual_viscosity
        
    def set_gas_properties(self, design_molecular_weight: float, actual_molecular_weight: float,
                          design_density: float = None, actual_density: float = None):
        """
        Set gas-specific properties.
        
        Args:
            design_molecular_weight (float): Design molecular weight (kg/kmol or lb/lbmol)
            actual_molecular_weight (float): Actual molecular weight (kg/kmol or lb/lbmol)
            design_density (float): Design density (kg/m³ or lb/ft³) - optional
            actual_density (float): Actual density (kg/m³ or lb/ft³) - optional
        """
        self.design_molecular_weight = design_molecular_weight
        self.actual_molecular_weight = actual_molecular_weight
        if design_density is not None:
            self.design_density = design_density
        if actual_density is not None:
            self.actual_density = actual_density
            
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
        """Display all flowmeter properties."""
        print(f"Flowmeter Type: {self.__class__.__name__}")
        print(f"Fluid Type: {self.fluid_type}")
        print(f"Design Pressure: {self.design_pressure}")
        print(f"Actual Pressure: {self.actual_pressure}")
        print(f"Design Temperature: {self.design_temperature}")
        print(f"Actual Temperature: {self.actual_temperature}")
        
        if self.fluid_type == "liquid":
            print(f"Design Density: {self.design_density}")
            print(f"Actual Density: {self.actual_density}")
            if self.design_viscosity:
                print(f"Design Viscosity: {self.design_viscosity}")
            if self.actual_viscosity:
                print(f"Actual Viscosity: {self.actual_viscosity}")
        elif self.fluid_type == "gas":
            print(f"Design Molecular Weight: {self.design_molecular_weight}")
            print(f"Actual Molecular Weight: {self.actual_molecular_weight}")
            if self.design_density:
                print(f"Design Density: {self.design_density}")
            if self.actual_density:
                print(f"Actual Density: {self.actual_density}")


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
            
            # For gases: ρ ∝ (P × MW) / T
            rho_design_ratio = (self.design_pressure * self.design_molecular_weight) / (self.design_temperature + 273.15)
            rho_actual_ratio = (self.actual_pressure * self.actual_molecular_weight) / (self.actual_temperature + 273.15)
            
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
            rho_design_ratio = (self.design_pressure * self.design_molecular_weight) / (self.design_temperature + 273.15)
            rho_actual_ratio = (self.actual_pressure * self.actual_molecular_weight) / (self.actual_temperature + 273.15)
            
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
            
            rho_design_ratio = (self.design_pressure * self.design_molecular_weight) / (self.design_temperature + 273.15)
            rho_actual_ratio = (self.actual_pressure * self.actual_molecular_weight) / (self.actual_temperature + 273.15)
            
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
            
            rho_design_ratio = (self.design_pressure * self.design_molecular_weight) / (self.design_temperature + 273.15)
            rho_actual_ratio = (self.actual_pressure * self.actual_molecular_weight) / (self.actual_temperature + 273.15)
            
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

if __name__ == "__main__":
    welcome_message()
