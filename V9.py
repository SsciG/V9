import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fastdtw import fastdtw
import logging
import os
import json
import sys




# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CupHandleDetector:
    """Advanced Cup and Handle pattern detection system prioritizing quality over quantity."""
    
    def __init__(self, config=None):
        """Initialize detector with configuration parameters."""
        # Default configuration with parameters adjusted for your dataset
        self.config = {
            # Extrema detection parameters
            "window_sizes": [2, 3, 4, 5, 6, 7, 8],   
            "prominence": 0.001,        
            "smooth_window": 1,           
            "smooth_poly": 1,            
            
            # Pattern geometry requirements
            "min_cup_depth": 0.0005,        
            "max_cup_depth": 1000,          
            "cup_symmetry_threshold": 0.45, 
            "max_handle_drop": 0.5,      
            "min_handle_drop": 0.001,       
            "handle_position_min": 0.3,   
             "min_cup_roundness": 0.5,
            # Duration constraints
            "min_cup_duration": 30,        
            "max_cup_duration": 14400,      
            "min_handle_duration": 15,    
            "max_handle_duration": 480,     
            "handle_to_cup_ratio_max": 0.9, 
            "rim_height_tolerance_pct": 0.05,
            # Quality scoring weights
            "weight_cup_depth": 0.2,
            "weight_cup_symmetry": 0.35,
            "weight_handle_position": 0.3,
            "weight_duration_ratio": 0.05,
            "weight_volume_pattern": 0.1,
            
            # Quality thresholds
            "min_quality_score": 45,       
            "confidence_threshold": 0.35,  
        }

        self.detection_stats = {
            'resistance_levels': {
                'total_found': 0,
                'top_levels': [],
                'price_range': None
            },

            
            'accumulation_analysis': {
                'levels_tested': 0,
                'valid_accumulations': 0,
                'rejection_reasons': {
                    'no_bars_in_zone': 0,
                    'periods_too_short': 0,
                    'high_volatility': 0,
                    'large_moves': 0,
                    'volatility_increase': 0
                }
            },
       

            
            'handle_analysis': {
                'accumulations_tested': 0,
                'handles_found': 0,
                'rejection_reasons': {
                    'insufficient_data': 0,
                    'depth_invalid': 0,
                    'duration_invalid': 0
                }
            },
            'pattern_creation': {
                'handles_tested': 0,
                'breakouts_found': 0,
                'patterns_created': 0
            },



        }

        self.config['debug'] = True
        self.config['verbose'] = True
        self.config['log_rejected_patterns'] = True
        self.deep_cup_tracking = {
            'potential_deep_cups': [],
            'deep_cup_rejections': [],
            'deep_cups_found': [],
            'counters': {
        'deep_cups_encountered': 0,
        'left_rim_attempts': 0,
        'left_rim_found_extrema': 0,
        'left_rim_found_local': 0,
        'left_rim_found_fallback': 0,
        'left_rim_total_failures': 0,
        'validation_failures_after_rim': 0,
        'patterns_created_deep': 0
            }
        }
        self.deep_cup_log = []
 
        # Update config with provided parameters if any
        if config:
            self.config.update(config)
            self.config.setdefault('enable_formation_first', False)
        
            print(f"\n🔧 LOADED CONFIG DEBUG:")
        print(f"  rim_height_tolerance_pct: {self.config.get('rim_height_tolerance_pct')}")
        print(f"  use_absolute_rim_tolerance: {self.config.get('use_absolute_rim_tolerance')}")
        print(f"  min_cup_depth: {self.config.get('min_cup_depth')}")
        print(f"  Config keys loaded: {list(self.config.keys())}")
        print(f"  Custom config passed: {self.config}")
        print(f"🔧 ACTUAL CONFIG: rim_tolerance={self.config['rim_height_tolerance_pct']}, min_depth={self.config['min_cup_depth']}")
        print("-" * 50)
     

        if hasattr(self, 'adaptive_config') and self.adaptive_config:
            # This will be set during detection when we have access to the dataframe
            pass  


    def calculate_atr(self, df, period=20):
        """Calculate Average True Range for volatility-based thresholds."""
        try:
            # Calculate True Range components
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            
            # True Range is the maximum of the three
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR is the rolling average of True Range
            atr = true_range.rolling(window=period, min_periods=1).mean()
            
            return atr
        except Exception as e:
            print(f'ATR calculation error: {e}')
            # Fallback: use simple high-low range
            return (df['high'] - df['low']).rolling(window=period).mean()
    
    def preprocess_data(self, df, price_col='close'):
        """Preprocess and clean data for pattern detection."""
        # Create a copy to avoid modifying original data
        processed = df.copy()
        
        # Ensure we have OHLC data
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in processed.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            # For missing columns, fill with available data
            for col in missing_cols:
                if col == 'open' and 'close' in processed.columns:
                    processed[col] = processed['close']
                elif col == 'high' and 'close' in processed.columns:
                    processed[col] = processed['close']
                elif col == 'low' and 'close' in processed.columns:
                    processed[col] = processed['close']
                elif col == 'close' and 'open' in processed.columns:
                    processed[col] = processed['open']
        
        # Apply Savitzky-Golay filter for smoothing if enough data points
        if len(processed) > self.config['smooth_window']:
            try:
                processed[f"{price_col}_smooth"] = savgol_filter(
                    processed[price_col].values, 
                    self.config['smooth_window'], 
                    self.config['smooth_poly']
                )
            except Exception as e:
                logger.warning(f"Smoothing error: {e}. Using original price data.")
                processed[f"{price_col}_smooth"] = processed[price_col]
        else:
            processed[f"{price_col}_smooth"] = processed[price_col]
        
        # Calculate log returns for volatility estimation
        processed['log_return'] = np.log(processed[price_col] / processed[price_col].shift(1))
        
        # Calculate rolling volatility
        if len(processed) > 20:
            processed['volatility'] = processed['log_return'].rolling(window=20).std()
            # Fill NaN values with mean volatility
            processed['volatility'] = processed['volatility'].fillna(processed['volatility'].mean())
        else:
            processed['volatility'] = 0.01  # Default volatility if data is too short
        
        return processed
    
    def detect_extrema_multi_scale(self, df, price_col='close_smooth'):
        """Detect peaks and troughs at multiple scales and combine results."""
        extrema_results = []
        
        # Apply each window size
        for window in self.config['window_sizes']:
            # Skip windows that are too large for the data
            if 2 * window + 1 >= len(df):
                continue
                
            # Adjust prominence based on local volatility
            local_volatility = df['volatility'].mean()
            adjusted_prominence = self.config['prominence'] * (1 + 10 * local_volatility)
            
            high_prices = df['high'].values
            low_prices = df['low'].values
            extrema = np.zeros(len(df))

            for i in range(window, len(df) - window):
                # Check for peaks using HIGH prices
                local_highs = high_prices[i - window:i + window + 1]
                center_high = high_prices[i]
                
                if center_high == max(local_highs):
                    if center_high > 0 and (center_high - np.min(local_highs)) / center_high >= adjusted_prominence:
                        extrema[i] = 1  # peak
                
                # Check for troughs using LOW prices  
                local_lows = low_prices[i - window:i + window + 1]
                center_low = low_prices[i]
                
                if center_low == min(local_lows):
                    if center_low > 0 and (np.max(local_lows) - center_low) / center_low >= adjusted_prominence:
                        extrema[i] = -1  # trou
                        
            # Store results for this window size
            extrema_results.append(extrema)
        
        # Combine results across window sizes
        if not extrema_results:
            # Return zeros if no valid window sizes
            return np.zeros(len(df))
            
        # Consensus approach: keep extrema only if detected in majority of scales
        extrema_combined = np.zeros(len(df))
        for i in range(len(df)):
            votes_peak = sum(1 for extrema in extrema_results if i < len(extrema) and extrema[i] == 1)
            votes_trough = sum(1 for extrema in extrema_results if i < len(extrema) and extrema[i] == -1)
            
            # Require majority agreement for peak or trough
            if votes_peak >= 1: # Just need 1 window to detect peak
                extrema_combined[i] = 1
            elif votes_trough >= 1:  # Just need 1 window to detect trough
                extrema_combined[i] = -1
        
        return extrema_combined
    
    def calculate_cup_symmetry(self, df, cup_start, cup_bottom, cup_end, price_col='close_smooth'):
        """Calculate symmetry score for cup formation."""
        # Extract the cup segment
        cup_segment = df.loc[cup_start:cup_end, price_col]
        
        # Find the index of cup bottom
        bottom_idx = df.index.get_loc(cup_bottom)
        
        # Calculate midpoint index
        start_idx = df.index.get_loc(cup_start)
        end_idx = df.index.get_loc(cup_end)
        
        # If cup bottom is not near the middle, cup is asymmetric
        left_half_length = bottom_idx - start_idx
        right_half_length = end_idx - bottom_idx
        
        # Calculate position symmetry (0-1)
        if max(left_half_length, right_half_length) == 0:
            position_symmetry = 0
        else:
            position_symmetry = min(left_half_length, right_half_length) / max(left_half_length, right_half_length)
        
        # Calculate shape symmetry using a simpler approach than DTW
        # (Due to the 1-D vector issues)
        try:
            # Get left and right halves
            left_half = df.loc[cup_start:cup_bottom, price_col].values
            right_half = df.loc[cup_bottom:cup_end, price_col].values
            
            # If either half is too short, cup is asymmetric
            if len(left_half) <= 1 or len(right_half) <= 1:
                return 0.5  # Default medium symmetry
            
            # Reverse the right half for comparison
            right_half_reversed = right_half[::-1]
            
            # Resize the longer half to match the shorter one
            if len(left_half) > len(right_half_reversed):
                # Downsample left half
                indices = np.linspace(0, len(left_half)-1, len(right_half_reversed)).astype(int)
                left_half = left_half[indices]
            elif len(right_half_reversed) > len(left_half):
                # Downsample right half
                indices = np.linspace(0, len(right_half_reversed)-1, len(left_half)).astype(int)
                right_half_reversed = right_half_reversed[indices]
            
            # Normalize both halves for comparison
            left_norm = (left_half - np.min(left_half)) / (np.max(left_half) - np.min(left_half) + 1e-10)
            right_norm = (right_half_reversed - np.min(right_half_reversed)) / (np.max(right_half_reversed) - np.min(right_half_reversed) + 1e-10)
            
            # Calculate mean absolute error
            mae = np.mean(np.abs(left_norm - right_norm))
            
            # Convert to similarity score (1 - normalized error)
            shape_symmetry = 1.0 - min(1.0, mae * 5)  # Scale error to get reasonable scores
            
        except Exception as e:
            logger.warning(f"Shape symmetry calculation error: {e}")
            shape_symmetry = 0.5  # Default medium symmetry
        
        # Combine position and shape symmetry
        combined_symmetry = 0.4 * position_symmetry + 0.6 * shape_symmetry
        
        return combined_symmetry
    

    
    
    
    def calculate_cup_roundness(self, df, cup_start, cup_bottom, cup_end, price_col='close_smooth'):
        """Enhanced roundness calculation that better identifies U-shapes vs V-shapes."""
        try:
            # Extract cup segment
            cup_segment = df.loc[cup_start:cup_end, price_col]
            if len(cup_segment) < 7:  # Need more points for U-shape analysis
                return 0.3  # Default medium roundness
            
            # Normalize x-axis to 0-1 range
            x = np.linspace(0, 1, len(cup_segment))
            y = cup_segment.values
            
            # Normalize y values
            min_val = np.min(y)
            max_val = np.max(y)
            range_val = max_val - min_val
            if range_val == 0:
                return 0  # Flat line, not a cup
                
            y_norm = (y - min_val) / range_val
            
            # 1. QUADRATIC FIT (U-shape detection)
            coeffs = np.polyfit(x, y_norm, 2)
            if coeffs[0] <= 0:  # Not a proper U-shape
                return 0.1
                
            poly = np.poly1d(coeffs)
            fitted_values = poly(x)
            ss_total = np.sum((y_norm - np.mean(y_norm))**2)
            ss_residual = np.sum((y_norm - fitted_values)**2)
            
            if ss_total == 0:
                r_squared = 0
            else:
                r_squared = 1 - (ss_residual / ss_total)
            
            # 2. BOTTOM ACCUMULATION ANALYSIS (Key for U-shapes)
            y_bottom_threshold = np.min(y_norm) + 0.15  # Bottom 15%
            bottom_bars = np.sum(y_norm <= y_bottom_threshold)
            bottom_ratio = bottom_bars / len(y_norm)

            if len(y_norm) >= 15:  # Only for longer cups
                # Look for sustained bottom periods (3+ consecutive bars)
                sustained_bottom_bars = 0
                consecutive_count = 0
                
                for val in y_norm:
                    if val <= y_bottom_threshold:
                        consecutive_count += 1
                        if consecutive_count >= 3:  # Sustained bottom period
                            sustained_bottom_bars += 1
                    else:
                        consecutive_count = 0
                
                # Bonus for sustained bottom accumulation (key U-shape characteristic)
                sustained_ratio = sustained_bottom_bars / len(y_norm)
                bottom_ratio = max(bottom_ratio, sustained_ratio * 1.5)  # Amplify sustained periods
                
            # 3. GRADIENT ANALYSIS (Smooth vs Sharp transitions)
            gradients = np.abs(np.diff(y_norm))
            max_gradient = np.max(gradients)
            avg_gradient = np.mean(gradients)
            gradient_smoothness = 1 - min(1.0, max_gradient / (avg_gradient + 1e-10) / 10)
            
            # 4. CURVATURE CONSISTENCY (U-shapes have consistent curvature)
            second_derivative = np.diff(y_norm, 2)
            curvature_consistency = 1 - np.std(second_derivative) / (np.abs(np.mean(second_derivative)) + 1e-10)
            curvature_consistency = max(0, min(1, curvature_consistency))
            
            # 5. COMPOSITE U-SHAPE SCORE
            base_roundness = max(0, min(1, r_squared))
            bottom_bonus = min(1.0, bottom_ratio * 4)  # Reward time at bottom
            smoothness_bonus = gradient_smoothness
            curvature_bonus = curvature_consistency
            
            # Enhanced composite score favoring U-shapes
            u_shape_score = (
                0.25 * base_roundness +      # Reduced weight on fit quality
                0.45 * bottom_bonus +        # INCREASED weight on bottom time (key for U-shapes)
                0.20 * smoothness_bonus +    # Keep smooth transitions
                0.10 * curvature_bonus       # Keep curvature consistency
            )

            # Add this bonus for genuine U-shapes
            if bottom_ratio > 0.25 and r_squared > 0.6:  # Strong U-shape indicators
                u_shape_score += 0.15  # 15% bonus for strong U-shapes

            return min(1.0, u_shape_score)
            
        except Exception as e:
            logger.warning(f"Error calculating enhanced cup roundness: {e}")
            return 0.3  # Default medium roundness
    


    def validate_rim_levels(self, peak_a_price, peak_c_price, breakout_e_price):
        """Ensure A, C, E are at similar resistance levels"""
        tolerance = self.config.get("rim_height_tolerance_pct", 2.5)
        resistance_level = max(peak_a_price, peak_c_price)
        
        # Check C alignment with A
        c_deviation = abs(peak_c_price - resistance_level) / resistance_level
        
        # Check E breakout above resistance  
        e_breakout = (breakout_e_price - resistance_level) / resistance_level
        
        return (c_deviation <= tolerance and 0 <= e_breakout <= 0.05)
    
    def get_adaptive_durations(self, timeframe_minutes):
        """Get duration limits based on timeframe - use config as base"""
        base_min_cup = self.config.get('min_cup_duration', 2880)
        base_max_cup = self.config.get('max_cup_duration', 14400)
        base_min_handle = self.config.get('min_handle_duration', 240)
        base_max_handle = self.config.get('max_handle_duration', 2880)
        
        # Apply multipliers based on timeframe
        if timeframe_minutes <= 15:      # 1-15 min data
            return {
                "min_cup_duration": int(base_min_cup * 0.5),    # Shorter for intraday
                "max_cup_duration": int(base_max_cup * 0.7),    
                "min_handle_duration": int(base_min_handle * 0.5),
                "max_handle_duration": int(base_max_handle * 0.7)
            }
        elif timeframe_minutes <= 60:    # 30-60 min data  
            return {
                "min_cup_duration": base_min_cup,
                "max_cup_duration": base_max_cup,
                "min_handle_duration": base_min_handle,
                "max_handle_duration": base_max_handle
            }
        else:                           # Daily+ data
            return {
                "min_cup_duration": int(base_min_cup * 2.5),    # Longer for daily
                "max_cup_duration": int(base_max_cup * 9),
                "min_handle_duration": int(base_min_handle * 6),
                "max_handle_duration": int(base_max_handle * 7.5)
            }


    def detect_cup_and_handle(self, df, extrema_col='extrema', price_col='close_smooth'):
        """By-the-book cup and handle detection"""
        
        patterns = []
        peaks = df[df[extrema_col] == 1]
        troughs = df[df[extrema_col] == -1]
        
        # For each potential left rim (Peak A)
        for i, peak_a_time in enumerate(peaks.index[:-1]):
            peak_a_price = df.loc[peak_a_time, 'high']
            
            # Find potential right rims (Peak C) 
            for peak_c_time in peaks.index[i+1:]:
                peak_c_price = df.loc[peak_c_time, 'high']
                
                # 1. Validate rim symmetry (your existing validation)
                if not self._validate_rim_symmetry(peak_a_price, peak_c_price):
                    continue
                    
                # 2. Find cup bottom between rims
                trough_b_time = self._find_cup_bottom(df, peak_a_time, peak_c_time, troughs)
                if not trough_b_time:
                    continue
                    
                # 3. Validate cup depth and shape (your existing validation)
                if not self._validate_cup_geometry(df, peak_a_time, trough_b_time, peak_c_time):
                    continue
                    
                # 4. Find handle after right rim (your existing method)
                handle = self._find_handle_after_cup(df, peak_c_time, peak_c_price)
                if not handle:
                    continue
                    
                # 5. Find breakout (your existing method)
                breakout_time = self._find_breakout(df, handle, max(peak_a_price, peak_c_price))
                if not breakout_time:
                    continue
                    
                # Create pattern using your existing quality scoring
                pattern = self._create_validated_pattern(df, peak_a_time, trough_b_time, 
                                                    peak_c_time, handle, breakout_time)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
        

    def detect_accumulation_base(self, df, resistance_level):
        """Find accumulation zones - QUIET with summary stats"""
        self.detection_stats['accumulation_analysis']['levels_tested'] += 1
        
        accumulation_floor = resistance_level * 0.75
        accumulation_ceiling = resistance_level * 0.98
        timeframe_minutes = self.detect_timeframe(df) if hasattr(self, 'detect_timeframe') else 15
        min_duration_minutes = self.config.get('min_cup_duration', 2880)  # Default from config
        min_duration_bars = max(3, min_duration_minutes // timeframe_minutes)

        print(f"🔧 FIXED LINE 603:")
        print(f"   Timeframe: {timeframe_minutes} minutes")
        print(f"   Config min_cup_duration: {min_duration_minutes} minutes")
        print(f"   Calculated min_duration_bars: {min_duration_bars}")
        
        # Quick check: any bars in zone?
        bars_in_zone = sum(1 for _, row in df.iterrows() 
                        if accumulation_floor <= row['close'] <= accumulation_ceiling)
        
        if bars_in_zone == 0:
            self.detection_stats['accumulation_analysis']['rejection_reasons']['no_bars_in_zone'] += 1
            return []
        
        accumulation_periods = []
        i = 0
        while i < len(df) - min_duration_bars:
            if accumulation_floor <= df.iloc[i]['close'] <= accumulation_ceiling:
                period_end = i
                while (period_end < len(df) and 
                    accumulation_floor <= df.iloc[period_end]['low'] and 
                    df.iloc[period_end]['high'] <= accumulation_ceiling):
                    period_end += 1
                
                period_length = period_end - i
                
                if period_length >= min_duration_bars:  # Use fixed min_duration_bars
                    period_data = df.iloc[i:period_end]
                    is_valid, rejection_reason = self.is_accumulation_period_summary(period_data)
                    
                    if is_valid:
                        accumulation_periods.append({
                            'start': period_data.index[0],
                            'end': period_data.index[-1],
                            'duration': period_length,
                            'avg_price': period_data['close'].mean(),
                            'volatility': period_data['close'].std() / period_data['close'].mean(),
                            'score': self.score_accumulation_quality(period_data)
                        })
                        self.detection_stats['accumulation_analysis']['valid_accumulations'] += 1
                    else:
                        self.detection_stats['accumulation_analysis']['rejection_reasons'][rejection_reason] += 1
                    
                    i = period_end
                else:
                    self.detection_stats['accumulation_analysis']['rejection_reasons']['periods_too_short'] += 1
                    i += 1
            else:
                i += 1
        
        return accumulation_periods
    
    def is_accumulation_period_summary(self, period_data):
        """Check accumulation - return (is_valid, rejection_reason)"""
        
        # Volatility check
        volatility = period_data['close'].std() / period_data['close'].mean()
        if volatility > 0.12:
            return False, 'high_volatility'
        
        # Large moves check
        max_move = abs(period_data['close'].pct_change()).max()
        if max_move > 0.15:
            return False, 'large_moves'
        
        # Volatility compression check
        if len(period_data) >= 6:
            first_half = period_data[:len(period_data)//2]
            second_half = period_data[len(period_data)//2:]
            
            first_vol = first_half['close'].std()
            second_vol = second_half['close'].std()
            
            if second_vol > first_vol * 1.5:
                return False, 'volatility_increase'
        
        return True, 'valid'
    
    

   

    def score_accumulation_quality(self, period_data):
        """Score accumulation quality - RELAXED scoring"""
        score = 0
        
        # Duration bonus - more generous
        if len(period_data) >= 12:  # Reduced from 30
            score += 25
        elif len(period_data) >= 8:   # Reduced from 20
            score += 15
        
        # Volatility bonus - more generous thresholds
        volatility = period_data['close'].std() / period_data['close'].mean()
        if volatility <= 0.04:       # Reduced from 2%
            score += 30
        elif volatility <= 0.08:     # Reduced from 3%
            score += 20
        elif volatility <= 0.12:     # New tier
            score += 10
        
        # Volume pattern (if available)
        if 'volume' in period_data.columns:
            price_median = period_data['close'].median()
            low_price_volume = period_data[period_data['close'] <= price_median]['volume'].mean()
            high_price_volume = period_data[period_data['close'] > price_median]['volume'].mean()
            
            if low_price_volume > high_price_volume:
                score += 25
        
        # Time consistency - more generous
        price_range = period_data['high'].max() - period_data['low'].min()
        avg_price = period_data['close'].mean()
        range_ratio = price_range / avg_price
        
        if range_ratio <= 0.15:      # Increased from 10% to 15%
            score += 20
        elif range_ratio <= 0.20:    # New tier
            score += 10
        
        return score
    
    def detect_resistance_levels(self, df, lookback_bars=None):
        """Find horizontal resistance levels - QUIET with summary stats"""

        if lookback_bars is None:
            timeframe_minutes = self.detect_timeframe(df)
            if timeframe_minutes == 15:
                lookback_bars = 24  # 6 hours for intraday patterns
            elif timeframe_minutes <= 60:
                lookback_bars = 48  # 12 hours  
            else:
                lookback_bars = 100  # Keep original for daily+
        
        print(f"🔍 RESISTANCE DETECTION: {lookback_bars} bars = {lookback_bars * timeframe_minutes / 60:.1f} hours")
        resistance_levels = []
        price_tolerance = df['close'].std() * 0.002
        
        # Store price range for summary
        self.detection_stats['resistance_levels']['price_range'] = {
            'min': df['low'].min(),
            'max': df['high'].max(),
            'tolerance': price_tolerance
        }
        
        for i in range(lookback_bars, len(df)):
            window = df.iloc[i-lookback_bars:i]
            high_prices = window['high'].values
            price_clusters = {}
            
            for price in high_prices:
                found_cluster = False
                for cluster_price in price_clusters:
                    if abs(price - cluster_price) <= price_tolerance:
                        price_clusters[cluster_price] += 1
                        found_cluster = True
                        break
                
                if not found_cluster:
                    price_clusters[price] = 1
            
            for level, touches in price_clusters.items():
                if touches >= 2:
                    resistance_levels.append({
                        'price': level,
                        'touches': touches,
                        'end_time': window.index[-1],
                        'strength': touches
                    })

                    
        
        # Sort and store summary
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)

        

        # Add price diversity - don't cluster similar resistance levels
        diverse_levels = []
        min_price_gap = df['close'].std() * 0.01  # 1% of price volatility

        for level in resistance_levels:
            # Check if this price level is too close to existing ones
            too_close = False
            for existing in diverse_levels:
                if abs(level['price'] - existing['price']) < min_price_gap:
                    too_close = True
                    break
            
            if not too_close:
                diverse_levels.append(level)
            
            if len(diverse_levels) >= 500:  # Increased from 20 to 100
                break
            
            max_process_bars = self.config.get('max_resistance_bars', 100000)
            if i > lookback_bars + max_process_bars:  # Limit processing to reasonable amount
                print(f"   ⚠️ Stopping resistance detection at {i} bars to prevent hanging")
                break

        unique_levels = diverse_levels
        
        self.detection_stats['resistance_levels']['total_found'] = len(resistance_levels)
        self.detection_stats['resistance_levels']['top_levels'] = [
            {'price': level['price'], 'touches': level['touches']} 
            for level in unique_levels[:5]
        ]
        
        return unique_levels
    
    def validate_pattern_duration(self, accumulation_start, accumulation_end, handle_end):
        """Validate realistic pattern durations for 4H timeframe"""
        
        # Cup duration (accumulation period)
        cup_duration_hours = (accumulation_end - accumulation_start).total_seconds() / 3600
        
        # For 4H ES futures, realistic cup durations:
        min_cup_minutes = self.config.get('min_cup_duration', 2880)  # Minutes from config
        max_cup_minutes = self.config.get('max_cup_duration', 14400)  # Minutes from config
        min_cup_hours = min_cup_minutes / 60
        max_cup_hours = max_cup_minutes / 60
        
        if cup_duration_hours > max_cup_hours:
            print(f"   ❌ REJECTED: Cup too long ({cup_duration_hours:.1f}h > {max_cup_hours:.1f}h)")
            return None
        
        if cup_duration_hours > max_cup_hours:
            return False, f"Cup too long: {cup_duration_hours:.1f}h > {max_cup_hours}h"
        
        # Total pattern duration
        total_duration_hours = (handle_end - accumulation_start).total_seconds() / 3600
        max_total_hours = 840  # 35 days total maximum
        
        if total_duration_hours > max_total_hours:
            return False, f"Pattern too long: {total_duration_hours:.1f}h > {max_total_hours}h"
        
        return True, f"Valid duration: cup={cup_duration_hours:.1f}h, total={total_duration_hours:.1f}h"
    
    def remove_duplicate_patterns(self, patterns):
        """Remove nearly identical patterns"""
        print(f"🔄 DEDUP: Processing {len(patterns)} patterns")
        unique_patterns = []
        
        for pattern in patterns:
            is_duplicate = False
            
            for existing in unique_patterns:
                # Check if same accumulation period (within 1 day)
                time_diff = abs((pattern['peak_a'] - existing['peak_a']).total_seconds())
                
                if time_diff < 86400:  # Less than 24 hours apart
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_patterns.append(pattern)
        
        return unique_patterns
    

    def validate_cup_shape(self, df, accumulation_start, accumulation_end):
        """Ensure there's an actual cup shape, not just accumulation"""
        
        cup_data = df.loc[accumulation_start:accumulation_end]
        
        # Must have clear descent and ascent
        start_price = cup_data['close'].iloc[0]
        end_price = cup_data['close'].iloc[-1]
        min_price = cup_data['low'].min()
        
        # Minimum cup depth (not just any accumulation)
        depth_from_start = (start_price - min_price) / start_price
        depth_from_end = (end_price - min_price) / end_price
        
        if depth_from_start < 0.008 or depth_from_end < 0.008:  # 3% minimum depth
            return False, "Insufficient cup depth for proper formation"
        
        return True, "Valid cup shape detected"
            
    def find_breakout_after_handle(self, df, handle_end, resistance_level, peak_a_price, peak_c_price, peak_c_time):
        """Find breakout - Enhanced with detailed debugging"""

        current_atr = self.calculate_atr(df, 20).iloc[-1]  # You already have this method
        breakout_threshold_min = peak_c_price + (current_atr * 0.5)
        breakout_threshold_max = peak_c_price + (current_atr * 0.75)
        print(f"   🔧 DEBUG: breakout_threshold initially set to ${breakout_threshold_min:.2f}")
        

        # DEBUG: Log all inputs
        print(f"\n🔍 BREAKOUT DEBUG:")
        print(f"   handle_end: {handle_end}")
        print(f"   resistance_level: ${resistance_level:.2f}")
        print(f"   rim_level: ${max(peak_a_price, peak_c_price):.2f}")
        
        self.detection_stats['pattern_creation']['handles_tested'] += 1
        
        try:
            search_start_idx = df.index.get_loc(handle_end) + 1
        except KeyError:
            print(f"   ❌ Handle end {handle_end} not found in index")
            return None
        
        if search_start_idx >= len(df):
            print(f"   ❌ Handle end at data boundary (idx {search_start_idx}/{len(df)})")
            return None
        
        # Calculate search window
        timeframe_minutes = self.detect_timeframe(df) if hasattr(self, 'detect_timeframe') else 15
        breakout_search_minutes = self.config.get('breakout_search_duration', 1440) # Default: 8 hours
        breakout_search_bars = max(20, breakout_search_minutes // timeframe_minutes)
        
        search_end_idx = min(len(df), search_start_idx + breakout_search_bars)
        search_bars = search_end_idx - search_start_idx
        
        print(f"   Search window: {search_bars} bars from {df.index[search_start_idx] if search_start_idx < len(df) else 'END'}")
        
        if search_bars <= 0:
            print(f"   ❌ No bars available for breakout search")
            return None
   
        # Search for breakout
        breakout_found = False
        breakout_time = None
        max_high_seen = 0
        
        print(f"   🔧 DEBUG: breakout_threshold_min before search: ${breakout_threshold_min:.2f}")
        print(f"   🔧 DEBUG: breakout_threshold_max before search: ${breakout_threshold_max:.2f}")
        
        for i in range(search_start_idx, search_end_idx):
            
            current_high = df.iloc[i]['high']
            max_high_seen = max(max_high_seen, current_high)
            
            # Simple breakout criteria: high >= threshold
            if breakout_threshold_min <= current_high <= breakout_threshold_max:
                breakout_found = True
                breakout_time = df.index[i]
                break

            print(f"   🔧 DEBUG: breakout_threshold_min before search: ${breakout_threshold_min:.2f}")
                    

        # if max_high_seen > rim_level:
        #     print(f"   ❌ INVALIDATED: Price exceeded rim during search (${max_high_seen:.2f} > ${rim_level:.2f})")
        #     return None
        
        if not breakout_found:
            print(f"   ❌ NO BREAKOUT: Max high seen ${max_high_seen:.2f} < threshold ${breakout_threshold_min:.2f}")
            print(f"   📊 Gap to breakout: ${breakout_threshold_min - max_high_seen:.2f} points")
            return None
        
        self.detection_stats['pattern_creation']['breakouts_found'] += 1
        return breakout_time

    
    def detect_handle_formation(self, df, resistance_level, peak_c_time, peak_c_price, peak_a_time, trough_b_time):
        """Find handles - FIXED: Use timeframe-aware durations"""
        self.detection_stats['handle_analysis']['accumulations_tested'] += 1
        
        peak_c_idx = df.index.get_loc(peak_c_time)
        handle_search_start = peak_c_idx + 1
        search_start_idx = peak_c_idx
        if search_start_idx >= len(df) - 5:
            self.detection_stats['handle_analysis']['rejection_reasons']['insufficient_data'] += 1
            return []
        print(f"         🔍 HANDLE DEBUG: peak_c_time={peak_c_time}, peak_c_price=${peak_c_price:.2f}")
        print(f"         🔍 HANDLE DEBUG: search_start_idx={search_start_idx}, df_length={len(df)}")
    
        # FIX: Replace hardcoded 40 with timeframe-aware calculation
        timeframe_minutes = self.detect_timeframe(df) if hasattr(self, 'detect_timeframe') else 15
        max_handle_duration_minutes = self.config.get('max_handle_duration', 2880)  # Default from config
        max_handle_bars = max(5, max_handle_duration_minutes // timeframe_minutes)
        
        print(f"🔧 HANDLE FORMATION FIX:")
        print(f"   Max handle duration: {max_handle_duration_minutes} minutes = {max_handle_bars} bars")
        
     
        search_window = df.iloc[search_start_idx:search_start_idx + max_handle_bars]
        handles = []
        
        # Get timeframe-appropriate duration limits
        min_handle_duration_minutes = self.config.get('min_handle_duration', 240)  # Default from config
        min_handle_bars = max(2, min_handle_duration_minutes // timeframe_minutes)
        max_handle_bars_inner = max(10, max_handle_duration_minutes // timeframe_minutes)
        timeframe_minutes = self.detect_timeframe(df) if hasattr(self, 'detect_timeframe') else 15
        min_handle_gap_bars = max(1, self.config.get('min_handle_gap_minutes', 90) // timeframe_minutes)

        best_handle = None
        best_score = 0
        handle_start_idx = min_handle_gap_bars

        for i in range(handle_start_idx + min_handle_bars, min(len(search_window), max_handle_bars_inner)):
            potential_handle = search_window.iloc[handle_start_idx:i]

            if len(potential_handle) < min_handle_bars:
                continue

            handle_low_price = potential_handle['low'].min()

            if handle_low_price >= peak_c_price:
                continue

            # STEP 1: Handle must be below rim (fundamental requirement)

            actual_rim_level = resistance_level  # This is the real resistance level
            if handle_low_price >= actual_rim_level * 0.998:  # Allow tiny tolerance for noise
                print(f"   ❌ Handle ABOVE resistance: ${handle_low_price:.2f} >= ${actual_rim_level:.2f}")
                continue
                    

            if potential_handle.index[0] <= peak_c_time:
                 continue  
            
            handle_start_time = potential_handle.index[0]
            if handle_start_time <= peak_c_time:
                continue  # Handle must be AFTER peak_c

            min_hours_after_bottom = 4 # Minimum 12 hours after cup bottom  
            min_handle_time = trough_b_time + pd.Timedelta(hours=min_hours_after_bottom)
            if handle_start_time < min_handle_time:
                continue  # Handle too close to cup bottom

            # VALIDATION 2: Handle must be in final recovery phase (upper 25% of cup)
            cup_total_duration = (peak_c_time - peak_a_time).total_seconds() / 60  # minutes
            min_handle_start_time = peak_a_time + pd.Timedelta(minutes=cup_total_duration * 0.60)
            if handle_start_time < min_handle_start_time:
                continue
                
            # Ensure meaningful time gap (at least 1 bar)
            time_gap_minutes = (handle_start_time - peak_c_time).total_seconds() / 60
            if time_gap_minutes < timeframe_minutes:
                continue 

            # STEP 2: Calculate pullback
            handle_depth_points = peak_c_price - handle_low_price
            handle_depth_pct = (handle_depth_points / peak_c_price) * 100

            # STEP 3: Minimum pullback requirement
            cup_peak_price = peak_c_price
            cup_bottom_price = df.loc[trough_b_time, 'low']
            handle_depth_valid, depth_msg = self.validate_handle_depth_ratio(cup_peak_price, cup_bottom_price, handle_low_price)
            if not handle_depth_valid:
                print(f"   ❌ Handle rejected: {depth_msg}")
                continue

            trough_b_price = df.loc[trough_b_time, 'low']
            pullback_depth = (actual_rim_level - handle_low_price) / actual_rim_level
            cup_depth = peak_c_price - trough_b_price
            min_pullback_points = max(0.25, cup_depth * 0.03)
            if (peak_c_price - handle_low_price) < min_pullback_points:
                print(f"   ❌ Handle too close to rim: {peak_c_price - handle_low_price:.2f} points < {min_pullback_points}")
                continue


            cup_peak_price = peak_c_price
            cup_bottom_price = df.loc[trough_b_time, 'low']
            handle_depth_valid, depth_msg = self.validate_handle_depth_ratio(cup_peak_price, cup_bottom_price, handle_low_price)
            if not handle_depth_valid:
                print(f"   ❌ Handle rejected: {depth_msg}")
                continue

            duration = len(potential_handle)
            
            # Use config-based depth validation
            # try:
            #     atr = self.calculate_atr(df, 20).iloc[-1]
            #     min_handle_drop_points = max(atr * 0.25, 1.5)
            #     min_handle_depth = min_handle_drop_points / peak_c_price
            # except:
            #     min_handle_depth = 0.001
                
            max_handle_depth = 0.40

            if (min_handle_bars <= duration <= max_handle_bars_inner):


                is_valid_rim, rim_reason = self.validate_handle_never_above_rim(
                    df, potential_handle.index[0], potential_handle.index[-1], peak_c_price
                )
                if not is_valid_rim:
                    continue

               
            
                
                handle_quality_score = self.score_handle_quality(potential_handle, peak_c_price, trough_b_price)
                if handle_quality_score > best_score:
                    best_score = handle_quality_score
                    best_handle = {
                        'start': potential_handle.index[0],
                        'end': potential_handle.index[-1],
                        'depth_pct': pullback_depth * 100,
                        'duration': duration,
                        'low_price': potential_handle['low'].min(),
                        'score': handle_quality_score
                    }

        # Only add the BEST handle found
        if best_handle:
            print(f"         ✅ BEST HANDLE FOUND: score={best_handle['score']}")
            handles.append(best_handle)
            self.detection_stats['handle_analysis']['handles_found'] += 1
        else:
            self.detection_stats['handle_analysis']['rejection_reasons']['depth_invalid'] += 1


        
        if not handles:
            print(f"         ❌ NO VALID HANDLE: No pullback found after gap")
            
        return handles
    
    def validate_handle_never_above_rim(self, df, handle_start, handle_end, rim_level):
        """CRITICAL: Handle must never exceed rim level"""
        
        handle_period = df.loc[handle_start:handle_end]
        max_handle_high = handle_period['high'].max()
        
        # Zero tolerance for handles above rim
        if max_handle_high > rim_level:
            return False, f"Handle violated rim: {max_handle_high} > {rim_level}"
        
        return True, "Handle respects rim level"
    
    def validate_handle_depth_ratio(self, cup_peak_price, cup_bottom_price, handle_low_price):
        """Validate that handle depth is proportionally shallow vs. the cup depth"""
        cup_depth = cup_peak_price - cup_bottom_price
        handle_depth = cup_peak_price - handle_low_price
        if cup_depth <= 0:
            return False, "Cup depth invalid (zero or negative)"
        
        depth_ratio = handle_depth / cup_depth
        
        if depth_ratio <= 0.25:
            return True, "OPTIMAL"
        elif depth_ratio <= 0.70:
            return True, "ACCEPTABLE"
        else:
            return False, f"INVALID: handle too deep ({depth_ratio:.2f} of cup depth)"
    
    def calculate_handle_position(self, cup_start, cup_end, handle_start):
        """Calculate where handle actually occurs in cup timeline"""
        
        total_cup_duration = (cup_end - cup_start).total_seconds()
        handle_offset = (handle_start - cup_start).total_seconds()
        
        position_ratio = handle_offset / total_cup_duration
        return min(1.0, max(0.0, position_ratio))
    

    def validate_right_rim_is_peak(self, df, peak_c_time):
        """Ensure right rim is an actual peak, not arbitrary point"""
        
        if peak_c_time not in df.index:
            return False
        
        # Must be marked as extrema peak
        if df.loc[peak_c_time, 'extrema'] != 1:
            return False
        
        return True

    def score_handle_quality(self, handle_data, resistance_level, trough_b_price=None):
        """Score handle formation quality - RELAXED for 4H ES"""
        score = 0
        
        # Depth scoring - more generous ranges
        handle_low = handle_data['low'].min()
        if trough_b_price is not None:
            # Use cup-proportional scoring (preferred)
            cup_depth = resistance_level - trough_b_price
            handle_depth = resistance_level - handle_low
            depth_ratio = handle_depth / cup_depth if cup_depth > 0 else 0
            depth_pct = depth_ratio * 100  # Convert to percentage for existing scoring logic
        else:
            # Fallback to old method
            depth_pct = (resistance_level - handle_low) / resistance_level * 100
        if 2 <= depth_pct <= 8:       # Optimal range (handles should be shallow!)
            score += 30
        elif 1 <= depth_pct <= 12:    # Acceptable range
            score += 20
        elif depth_pct <= 15:         # Still acceptable  
            score += 10
                
        # Duration scoring - more generous
        if 6 <= len(handle_data) <= 15:   # Optimal
            score += 25
        elif 3 <= len(handle_data) <= 20: # Acceptable
            score += 15
        elif len(handle_data) <= 25:      # Still acceptable
            score += 10
        
        # Controlled decline - more lenient for 4H gaps
        max_decline = abs(handle_data['close'].pct_change()).max()
        if max_decline <= 0.08:        # Increased from 5%
            score += 25
        elif max_decline <= 0.12:      # Increased from 8%
            score += 15
        elif max_decline <= 0.18:      # New tier for 4H tolerance
            score += 10
        
        # Volume decrease (if available)
        if 'volume' in handle_data.columns and len(handle_data) >= 3:
            early_vol = handle_data[:len(handle_data)//2]['volume'].mean()
            late_vol = handle_data[len(handle_data)//2:]['volume'].mean()
            if late_vol < early_vol:
                score += 20
        
        return score
    
    def validate_cup_shape_geometry(self, df, peak_a_time, trough_b_time, peak_c_time, timeframe_minutes=15):
        """Validate that the formation actually looks like a cup, not just any price decline."""
        try:
            # ADD THIS: Calculate rough_depth for adaptive validation
            peak_a_price = df.loc[peak_a_time, 'high']
            peak_c_price = df.loc[peak_c_time, 'high']
            trough_b_price = df.loc[trough_b_time, 'low']
            resistance_level = max(peak_a_price, peak_c_price)
            rough_depth = (resistance_level - trough_b_price) / resistance_level * 100
            
            # Get the cup segment
            cup_data = df.loc[peak_a_time:peak_c_time]
            
            if len(cup_data) < 10:
                return False, "Cup too short for geometric analysis"
            
            # 1. CHECK: Descent should be gradual, not a cliff drop
            peak_a_idx = df.index.get_loc(peak_a_time)
            trough_b_idx = df.index.get_loc(trough_b_time)
            
            descent_period = df.iloc[peak_a_idx:trough_b_idx + 1]
            if len(descent_period) >= 3:
                # Adaptive cliff drop tolerance based on cup depth
                cliff_tolerance = -0.05  # Default 5%
                if rough_depth >= 15.0:  # Very deep cups (15%+)
                    cliff_tolerance = -0.12  # Allow 12% drops
                elif rough_depth >= 10.0:  # Deep cups (10-15%)
                    cliff_tolerance = -0.08  # Allow 8% drops
                elif rough_depth >= 8.0:   # Medium-deep cups (8-10%)
                    cliff_tolerance = -0.06  # Allow 6% drops

                max_single_drop = descent_period['low'].pct_change().min()
                if max_single_drop < cliff_tolerance:
                    return False, f"Cliff drop detected: {max_single_drop:.1%} in single bar (limit: {cliff_tolerance:.1%})"
            
            # 2. CHECK: Bottom should be sustained, not a spike
            # Define "bottom zone" as lowest 20% of cup
            cup_height = max(peak_a_price, peak_c_price) - trough_b_price
            bottom_threshold = trough_b_price + (cup_height * 0.2)
            
            # Count bars in bottom zone
            bottom_bars = len(cup_data[cup_data['low'] <= bottom_threshold])
            bottom_percentage = bottom_bars / len(cup_data)
            
            # Adaptive bottom time requirement based on cup depth
            min_bottom_time = 0.05  # Default 15%
            if rough_depth >= 15.0:     # Very deep cups
                min_bottom_time = 0.08  # Only 8% time needed
            elif rough_depth >= 10.0:   # Deep cups  
                min_bottom_time = 0.10  # Only 10% time needed
            elif rough_depth >= 8.0:    # Medium-deep cups
                min_bottom_time = 0.12  # Only 12% time needed

            if bottom_percentage < min_bottom_time and rough_depth < 8.0: 
                return False, f"Spike reversal - only {bottom_percentage:.1%} time near bottom (need: {min_bottom_time:.1%})"
            
            # 3. CHECK: Should be roughly U or V shaped, not irregular
            # Simple trend check: price should generally rise after the bottom
            post_bottom_idx = trough_b_idx
            recovery_period = df.iloc[post_bottom_idx:df.index.get_loc(peak_c_time) + 1]
            
            if len(recovery_period) >= 3:
                # Check if recovery is generally upward
                start_recovery = recovery_period['close'].iloc[0]
                end_recovery = recovery_period['close'].iloc[-1]
                
                min_recovery = 1.005 if timeframe_minutes <= 15 else 1.02  # Less than 2% recovery
                if end_recovery <= start_recovery * min_recovery:
                    return False, f"No recovery after bottom - not a cup shape"
            
            return True, f"Valid cup geometry - {bottom_percentage:.1%} time at bottom"
            
        except Exception as e:
            return False, f"Cup geometry validation error: {str(e)}"
            

    def log_deep_cup_rejection(self, cup_info, reason, stage):
        """Log deep cup rejections to file for analysis"""
        if not hasattr(self, 'deep_cup_log'):
            self.deep_cup_log = []
        
        log_entry = f"DEEP CUP REJECTED: {cup_info['estimated_depth']:.1f}% depth - {reason}"
        log_entry += f" | Stage: {stage} | Period: {cup_info['accumulation_start']} to {cup_info['accumulation_end']}"
        self.deep_cup_log.append(log_entry)
        
        # Write to file immediately
        with open('deep_cup_rejections.log', 'a') as f:
            f.write(f"{pd.Timestamp.now()}: {log_entry}\n")


  

    def create_institutional_pattern(self, resistance, accumulation, handle, breakout_time, df,actual_peak_c_time, price_col='close_smooth'):
        """Simplified pattern creation - clean and working version."""
        print(f"\n🔧 CREATING PATTERN - Resistance: ${resistance['price']:.2f}")
        if breakout_time is None:
            return None
        
        # Step 1: Find cup bottom (lowest point in accumulation period)
        cup_data = df.loc[accumulation['start']:accumulation['end']]
        trough_b_idx = cup_data['low'].idxmin()
        trough_b_price = df.loc[trough_b_idx, 'low']
        
        # Step 2: Use accumulation start/end as rim points for now
        peak_a_time = accumulation['start']
        peak_a_price = df.loc[peak_a_time, 'high']
        peak_c_time = actual_peak_c_time 
        peak_c_price = df.loc[peak_c_time, 'high']


        rim_diff_abs = abs(peak_a_price - peak_c_price)
        rim_diff_pct = (rim_diff_abs / max(peak_a_price, peak_c_price)) * 100
        rim_level = (peak_a_price + peak_c_price) / 2
        
        print(f"   Cup: A=${peak_a_price:.2f} → B=${trough_b_price:.2f} → C=${peak_c_price:.2f}")
        print(f"   🎯 RIM ANALYSIS: A=${peak_a_price:.2f}, C=${peak_c_price:.2f}, Diff={rim_diff_pct:.2f}%")
    
        
        # Step 3: Calculate cup metrics
        cup_depth = peak_a_price - trough_b_price
        cup_depth_pct = (cup_depth / peak_a_price) * 100
        cup_duration_hours = (peak_c_time - peak_a_time).total_seconds() / 3600
        
        # Step 4: Calculate handle metrics  
        handle_low_price = df.loc[handle['start']:handle['end']]['low'].min()
        current_atr = self.calculate_atr(df, 20).iloc[-1] 
     
        handle_depth = peak_c_price - handle_low_price
        if handle_low_price > peak_c_price:
            print(f"   🚨 CRITICAL ERROR: Handle low ${handle_low_price:.2f} ABOVE rim ${peak_c_price:.2f}")
            print(f"   This should have been caught in handle formation validation!")
            return None
        cup_depth = peak_c_price - trough_b_price
        handle_depth_pct_of_rim = (handle_depth / peak_c_price) * 100  # Keep for reference
        handle_depth_pct = (handle_depth / cup_depth) * 100
        handle_duration_hours = (handle['end'] - handle['start']).total_seconds() / 3600
        depth_ratio = handle_depth / cup_depth

        if handle_depth <= 0:
            print(f"   ❌ Invalid handle: depth={handle_depth:.2f} (handle above rim)")
            return None
                
        print(f"   Handle: depth={handle_depth_pct:.2f}%, duration={handle_duration_hours:.1f}h")
        
        # Step 5: Basic validation
        if cup_depth_pct < 0.25:  
            print(f"   ❌ Cup too shallow: {cup_depth_pct:.2f}%")
            return None
            
        if handle_depth_pct > 100.0: 
            print(f"   ❌ Handle too deep: {handle_depth_pct:.2f}%")
            return None
        
        # Step 6: Calculate quality metrics
        cup_symmetry = self.calculate_cup_symmetry(df, peak_a_time, trough_b_idx, peak_c_time, price_col)
        cup_roundness = self.calculate_cup_roundness(df, peak_a_time, trough_b_idx, peak_c_time, price_col)
        quality_score = 50 + (cup_depth_pct * 2) + (cup_symmetry * 30) + (cup_roundness * 20)
        
        # Step 7: Create pattern object
        pattern = {
            'peak_a': peak_a_time,
            'trough_b': trough_b_idx, 
            'peak_c': peak_c_time,
            'handle_d': handle['start'],
            'breakout_e': breakout_time,
            'breakout_threshold': max(peak_a_price, peak_c_price),
            'breakout_confirmed': True,
            'rim_a_price': peak_a_price,           # Actual left rim price
            'rim_c_price': peak_c_price,           # Actual right rim price  
            'rim_level': rim_level,                # Average rim level
            'rim_diff_abs': rim_diff_abs,          # Absolute price difference
            'rim_diff_pct': rim_diff_pct,          # Percentage difference
            'rim_tolerance_used': self.config.get('rim_height_tolerance_pct', 'unknown'),  # What tolerance was used
            'cup_depth': cup_depth,
            'cup_depth_pct': cup_depth_pct,
            'cup_duration_min': cup_duration_hours * 60,
            'cup_symmetry': cup_symmetry,
            'cup_roundness': cup_roundness,
            'handle_depth': handle_depth,
            'handle_depth_pct': handle_depth_pct,
            'handle_duration_min': handle_duration_hours >= 60,
            'handle_position': 0.7,
            'handle_depth_ratio_to_cup': (peak_c_price - handle_low_price) / (peak_c_price - trough_b_price),
            'handle_to_cup_ratio': handle_duration_hours / cup_duration_hours if cup_duration_hours > 0 else 0,
            'quality_score': quality_score,
            'confidence_score': 0.8,
            'valid': True,
            'resistance_touches': resistance['touches'],
            'accumulation_score': accumulation['score'],
            'handle_score': handle['score'],
            'detection_method': 'simplified_institutional'
        }
        
        self.detection_stats['pattern_creation']['patterns_created'] += 1
        print(f"   ✅ PATTERN CREATED: Quality={quality_score:.1f}")
        return pattern
    
    def is_significant_peak(self, df, peak_time, atr_multiplier=1.5):
        """Check if peak is significant enough to be a rim (using ATR)"""
        try:
            # Get ATR at this point in time
            atr_series = self.calculate_atr(df, 20)
            peak_idx = df.index.get_loc(peak_time)
            current_atr = atr_series.iloc[peak_idx] if peak_idx < len(atr_series) else atr_series.iloc[-1]
            
            # Get peak price and surrounding prices
            peak_price = df.loc[peak_time, 'high']
            
            # Look at 5 bars before and after (or available range)
            start_idx = max(0, peak_idx - 5)
            end_idx = min(len(df), peak_idx + 6)
            local_window = df.iloc[start_idx:end_idx]
            
            # Peak must exceed surrounding area by at least 1.5x ATR
            min_surrounding = local_window['low'].min()
            peak_prominence = peak_price - min_surrounding
            required_prominence = current_atr * atr_multiplier
            
            is_significant = peak_prominence >= required_prominence
            
            print(f"    🔍 Peak {peak_time}: prominence={peak_prominence:.2f}, required={required_prominence:.2f}, significant={is_significant}")
            
            return is_significant
            
        except Exception as e:
            print(f"    ⚠️ ATR significance check failed: {e}")
            return True  # Default to accepting if check fails
    

    def find_optimal_right_rim(self, df, accumulation_start, accumulation_end, resistance_level, left_rim_price, trough_time):
        """Find actual valid right rim - must be real peak at end of cup recovery"""
        
        try:
            # Step 1: Your existing setup code (UNCHANGED)
            trough_idx = df.index.get_loc(trough_time)
            acc_end_idx = df.index.get_loc(accumulation_end)
            
            recovery_start = trough_idx + int((acc_end_idx - trough_idx) * 0.75)
            search_end = min(len(df), acc_end_idx + 20)
            
            if recovery_start >= search_end or recovery_start >= len(df):
                print(f"      ❌ Invalid search range, using accumulation_end")
                return accumulation_end
            
            search_window = df.iloc[recovery_start:search_end]
            actual_peaks = search_window[search_window['extrema'] == 1]
            
            if len(actual_peaks) == 0:
                print(f"      ❌ No actual peaks found in recovery phase, using accumulation_end")
                return accumulation_end  # ← ALWAYS VALID FALLBACK
            
            use_hybrid = getattr(self, 'use_hybrid_detection', False)
            print(f"\n      🔍 DEBUG HYBRID DETECTION:")
            print(f"         Hybrid mode enabled: {use_hybrid}")
            print(f"         Total peaks found: {len(actual_peaks)}")
        
            if use_hybrid:
                print(f"      🔄 Using hybrid detection (ATR + Direct scoring)")
                
                # Method 1: ATR filtering (if available)
                atr_candidates = []
                has_atr_method = hasattr(self, 'is_significant_peak')
                print(f"         Has ATR method: {has_atr_method}")

                if hasattr(self, 'is_significant_peak'):
                    for peak_time, peak_row in actual_peaks.iterrows():
                        if self.is_significant_peak(df, peak_time, atr_multiplier=1.5):
                            atr_candidates.append((peak_time, peak_row))
                    print(f"      📊 ATR Filter: {len(actual_peaks)} → {len(atr_candidates)} significant peaks")
                
                # Method 2: Use all peaks if ATR found none, or if ATR not available
                if not atr_candidates:
                    atr_candidates = [(peak_time, peak_row) for peak_time, peak_row in actual_peaks.iterrows()]
                    print(f"      📊 Using all {len(atr_candidates)} peaks (ATR filter not available/found none)")
            else:
                # Original behavior - use all peaks
                atr_candidates = [(peak_time, peak_row) for peak_time, peak_row in actual_peaks.iterrows()]
            
            # Step 3: Filter for ATR-significant peaks only
            significant_peaks = []
            for peak_time, peak_row in actual_peaks.iterrows():
                if self.is_significant_peak(df, peak_time, atr_multiplier=1.5):
                    significant_peaks.append((peak_time, peak_row))
            
            if not significant_peaks:
                print(f"      ❌ No ATR-significant peaks found, using best available")
                significant_peaks = [(peak_time, peak_row) for peak_time, peak_row in actual_peaks.iterrows()]
            
            print(f"      📊 ATR Filter: {len(actual_peaks)} peaks → {len(significant_peaks)} significant")
            
            # Step 3: Score peaks based on multiple criteria
            candidates = []
            for peak_time, peak_row in atr_candidates:  # ← ONLY CHANGE: use atr_candidates instead of actual_peaks.iterrows()
                peak_price = peak_row['high']
                
                # Your existing scoring logic (UNCHANGED)
                price_diff_pct = abs(peak_price - left_rim_price) / left_rim_price * 100
                symmetry_score = max(0, 100 - price_diff_pct * 40)
                
                position_in_recovery = (df.index.get_loc(peak_time) - recovery_start) / (search_end - recovery_start)
                position_score = position_in_recovery * 100
                
                peak_idx = df.index.get_loc(peak_time)
                lookback = min(5, peak_idx)
                lookahead = min(5, len(df) - peak_idx - 1)
                
                if lookback > 0 and lookahead > 0:
                    local_window = df.iloc[max(0, peak_idx-lookback):min(len(df), peak_idx+lookahead+1)]
                    is_local_high = peak_price >= local_window['high'].max()
                    strength_score = 100 if is_local_high else 50
                else:
                    strength_score = 50
                
                total_score = (
                    0.8 * symmetry_score +   
                    0.1 * position_score +    
                    0.1 * strength_score       
                )
                
                candidates.append({
                    'time': peak_time,
                    'price': peak_price,
                    'score': total_score,
                    'symmetry_pct': price_diff_pct,
                    'is_strong_peak': is_local_high if 'is_local_high' in locals() else False
                })
            
            # Rest of your existing code (UNCHANGED)
            if not candidates:
                print(f"      ❌ No valid candidates, using accumulation_end")
                return accumulation_end
            
            best_candidate = max(candidates, key=lambda x: x['score'])
            
            if best_candidate['symmetry_pct'] > 5.0:
                print(f"      ❌ Best candidate has {best_candidate['symmetry_pct']:.1f}% asymmetry, using accumulation_end")
                return accumulation_end
            
            result_time = best_candidate['time']
            if result_time not in df.index:
                print(f"      ❌ Result time {result_time} not in df.index, using accumulation_end")
                return accumulation_end
            
            print(f"      ✅ VALID RIGHT RIM: ${best_candidate['price']:.2f} (symmetry: {best_candidate['symmetry_pct']:.1f}%)")
            return result_time
            
        except Exception as e:
            print(f"      ❌ Error in find_optimal_right_rim: {e}, using accumulation_end")
            return accumulation_end
        

    def _score_handle_optimally(self, handle_depth_pct):
        """Score handle depth with optimal ranges"""
        if 3 <= handle_depth_pct <= 15:  # Optimal range
            return 20
        elif 1 <= handle_depth_pct <= 25:  # Acceptable range
            return 15
        elif handle_depth_pct <= 33:  # Still valid
            return 10
        else:
            return 5
        

    def validate_no_higher_highs_in_cup(self, df, peak_a_time, peak_c_time, tolerance_pct=2.0):
        """
        Validate that price doesn't go significantly above rim level during cup formation.
        This prevents invalid "cups" where price actually went higher between the rims.
        """
        try:
            # Get rim prices
            peak_a_price = df.loc[peak_a_time, 'high']
            peak_c_price = df.loc[peak_c_time, 'high'] 
            max_rim_price = max(peak_a_price, peak_c_price)
            
            # Define violation threshold (2% above max rim by default)
            violation_threshold = max_rim_price * (1 + tolerance_pct/100)
            
            # Check entire cup period
            cup_period = df.loc[peak_a_time:peak_c_time]
            
            # Find highest point in cup period
            max_high_in_cup = cup_period['high'].max()
            max_high_time = cup_period['high'].idxmax()
            
            # Calculate violation
            violation_amount = max_high_in_cup - max_rim_price
            violation_pct = (violation_amount / max_rim_price) * 100
            
            if violation_pct > tolerance_pct:
                return False, f"Invalid cup: price reached ${max_high_in_cup:.2f} at {max_high_time} ({violation_pct:.1f}% above rim ${max_rim_price:.2f})"
            
            return True, f"Valid cup: max high ${max_high_in_cup:.2f} within {tolerance_pct}% of rim ${max_rim_price:.2f}"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _score_duration_balance(self, cup_hours, handle_hours):
        """Reward balanced duration ratios"""
        if cup_hours <= 0 or handle_hours <= 0:
            return 0
        
        ratio = cup_hours / handle_hours
        if 3 <= ratio <= 8:  # Ideal cup-to-handle ratio
            return 15
        elif 2 <= ratio <= 12:  # Acceptable
            return 10
        else:
            return 5

    
    def validate_rim_heights_professional(self, peak_a_price, peak_c_price):
        """Professional-grade rim height validation"""
        
        # Calculate both absolute and percentage differences
        abs_diff = abs(peak_a_price - peak_c_price)
        pct_diff = abs_diff / max(peak_a_price, peak_c_price) * 100
        
        # Use different tolerances based on price level
        if max(peak_a_price, peak_c_price) < 50:  # Low-priced securities
            tolerance = 0.5  # 0.5% tolerance
        elif max(peak_a_price, peak_c_price) < 500:  # Mid-priced
            tolerance = 0.3  # 0.3% tolerance  
        else:  # High-priced
            tolerance = 0.2  # 0.2% tolerance
        
        return pct_diff <= tolerance
    
    def validate_pattern_context(self, pattern, df_daily=None):
        """Validate pattern against higher timeframe context"""
        if df_daily is None:
            return True
        
        # Ensure daily trend supports the pattern
        pattern_start = pattern['peak_a']
        pattern_end = pattern['breakout_e']
        
        daily_data = df_daily.loc[pattern_start:pattern_end]
        
        # Check if we're in a proper daily uptrend or consolidation
        daily_trend = daily_data['close'].iloc[-1] / daily_data['close'].iloc[0]
        
        # Pattern should form in context of larger uptrend (>5% gain over period)
        return daily_trend >= 1.05
    
    def validate_volume_breakout(self, df, handle_end, resistance_level):
        """Ensure breakout has proper volume confirmation"""
        search_start = df.index.get_loc(handle_end) + 1
        
        # Get average volume during handle formation
        handle_start = handle_end - pd.Timedelta(days=3)  # Adjust based on timeframe
        handle_volume = df.loc[handle_start:handle_end]['volume'].mean()
        
        # Look for breakout with 1.5x handle volume
        for i in range(search_start, min(len(df), search_start + 10)):
            if (df.iloc[i]['high'] >= resistance_level and 
                df.iloc[i]['volume'] >= handle_volume * 1.5):
                return df.index[i]
        
        return None
    
    

    
    def find_lowest_point_in_period(self, df, start_time, end_time):
        """Find the actual lowest point during accumulation period"""
        period_data = df.loc[start_time:end_time]
        if len(period_data) == 0:
            return start_time
        
        lowest_idx = period_data['low'].idxmin()
        return lowest_idx

    def print_detection_summary(self):
        """Print comprehensive detection summary at the end"""
        print(f"\n" + "="*80)
        print(f"🏛️  INSTITUTIONAL DETECTION DETAILED SUMMARY")
        print(f"="*80)
        
        # Resistance Analysis
        r_stats = self.detection_stats['resistance_levels']
        print(f"\n📊 RESISTANCE LEVEL ANALYSIS:")
        print(f"   Total levels found: {r_stats['total_found']}")
        if r_stats['price_range']:
            pr = r_stats['price_range']
            print(f"   Price range: ${pr['min']:.2f} - ${pr['max']:.2f}")
            print(f"   Price tolerance: ${pr['tolerance']:.2f}")
        
        if r_stats['top_levels']:
            print(f"   🔝 Top 5 resistance levels:")
            for i, level in enumerate(r_stats['top_levels']):
                print(f"      #{i+1}: ${level['price']:.2f} ({level['touches']} touches)")
        
        # Accumulation Analysis
        a_stats = self.detection_stats['accumulation_analysis']
        print(f"\n🏗️  ACCUMULATION ANALYSIS:")
        print(f"   Resistance levels tested: {a_stats['levels_tested']}")
        print(f"   Valid accumulations found: {a_stats['valid_accumulations']}")
        
        if a_stats['levels_tested'] > 0:
            success_rate = (a_stats['valid_accumulations'] / a_stats['levels_tested']) * 100
            print(f"   Success rate: {success_rate:.1f}%")
        
        print(f"   ❌ Rejection breakdown:")
        for reason, count in a_stats['rejection_reasons'].items():
            if count > 0:
                print(f"      {reason.replace('_', ' ').title()}: {count}")
        
        # Handle Analysis
        h_stats = self.detection_stats['handle_analysis']
        print(f"\n🔧 HANDLE ANALYSIS:")
        print(f"   Accumulations tested: {h_stats['accumulations_tested']}")
        print(f"   Handles found: {h_stats['handles_found']}")
        
        if h_stats['accumulations_tested'] > 0:
            success_rate = (h_stats['handles_found'] / h_stats['accumulations_tested']) * 100
            print(f"   Success rate: {success_rate:.1f}%")
        
        print(f"   ❌ Rejection breakdown:")
        for reason, count in h_stats['rejection_reasons'].items():
            if count > 0:
                print(f"      {reason.replace('_', ' ').title()}: {count}")
        
        # Pattern Creation
        p_stats = self.detection_stats['pattern_creation']
        print(f"\n🎯 PATTERN CREATION:")
        print(f"   Handles tested for breakout: {p_stats['handles_tested']}")
        print(f"   Breakouts found: {p_stats['breakouts_found']}")
        print(f"   Final patterns created: {p_stats['patterns_created']}")
        
        if p_stats['handles_tested'] > 0:
            breakout_rate = (p_stats['breakouts_found'] / p_stats['handles_tested']) * 100
            print(f"   Breakout success rate: {breakout_rate:.1f}%")
        
        # Overall Summary
        print(f"\n🎯 OVERALL PIPELINE:")
        print(f"   {r_stats['total_found']} resistance levels → "
            f"{a_stats['valid_accumulations']} accumulations → "
            f"{h_stats['handles_found']} handles → "
            f"{p_stats['breakouts_found']} breakouts → "
            f"{p_stats['patterns_created']} patterns")
        
        print(f"="*80) 


    def detect_u_shaped_patterns(self, df, existing_patterns):
        """
        Secondary detection focused on finding U-shaped cups that might have been missed.
        This runs AFTER the main detection to supplement results.
        """
        
        # Create U-shape focused configuration
        u_shape_config = self.config.copy()
        u_shape_config.update({
    # Build on YOUR config values, just adjust for U-shapes
    "min_cup_roundness": max(0.75, self.config.get('min_cup_roundness', 0.1)),  # At least 0.75 for U-shapes
    "rim_height_tolerance_pct": max(self.config.get('rim_height_tolerance_pct', 5.0), 1.5),  # Use your tolerance or higher
    "min_quality_score": min(self.config.get('min_quality_score', 20), 30),  # Use your threshold or lower
    "min_cup_to_handle_ratio": self.config.get('min_cup_to_handle_ratio', 0.3),  # Use YOUR ratio
})
        
        # Temporarily update config
        original_config = self.config
        self.config = u_shape_config
        
        print(f"\n🔄 RUNNING U-SHAPE FOCUSED DETECTION...")
        print(f"   Target: High roundness (≥0.75) and symmetry (≥0.2)")
        print(f"   Relaxed: Rim tolerance (1.5%), Quality (30)")
        
        # Run detection with U-shape focus
        u_patterns, u_rejections = self.detect_cup_and_handle(df, 'extrema', 'close_smooth')
        
        # Restore original config
        self.config = original_config
        
        # Filter out any patterns that overlap with existing ones
        new_patterns = []
        existing_times = [(p['peak_a'], p['breakout_e']) for p in existing_patterns]
        
        for pattern in u_patterns:
            pattern_time = (pattern['peak_a'], pattern['breakout_e'])
            
            # Check for time overlap with existing patterns
            overlaps = False
            for existing_start, existing_end in existing_times:
                # Check if patterns overlap in time
                if (pattern_time[0] <= existing_end and pattern_time[1] >= existing_start):
                    overlaps = True
                    break
            
            # Only add if it's a true U-shape and doesn't overlap
            if not overlaps and pattern['cup_roundness'] >= 0.75:
                pattern['detection_type'] = 'u_shape_focused'
                new_patterns.append(pattern)
                print(f"   ✅ Found NEW U-shape: {pattern['peak_a']} (roundness: {pattern['cup_roundness']:.2f})")
        
        print(f"   📊 U-Shape Detection Results: {len(new_patterns)} new patterns")
        
        return new_patterns, u_rejections
    
    def filter_patterns(self, patterns):
        """Filter patterns based on quality score."""
        print(f"🔍 FILTERING DEBUG:")
        print(f"  Input patterns: {len(patterns)}")
        print(f"  Min quality threshold: {self.config['min_quality_score']}")
        
        # STEP 1: Remove incomplete patterns FIRST
        complete_patterns = []
        required_fields = ['peak_a', 'trough_b', 'peak_c', 'handle_d', 'breakout_e']
        
        for i, p in enumerate(patterns):
            # Check for None values in critical fields
            missing_fields = [field for field in required_fields if p.get(field) is None]
            if missing_fields:
                print(f"   ❌ Pattern {i+1} missing: {missing_fields}")
                continue
            
            # Check if timestamps can be converted safely
            try:
                if pd.to_datetime(p['breakout_e']) is None or pd.to_datetime(p['peak_a']) is None:
                    print(f"   ❌ Pattern {i+1} has invalid timestamps")
                    continue
            except:
                print(f"   ❌ Pattern {i+1} timestamp conversion failed")
                continue
                
            complete_patterns.append(p)
        
        print(f"  Complete patterns after validation: {len(complete_patterns)}")
        
        # STEP 2: Continue with quality filtering on complete patterns only
        if hasattr(self, '_processed_df') and self._processed_df is not None:
            recent_volatility = self._processed_df['volatility'].tail(100).mean()
        else:
            recent_volatility = 0.02
        volatility_multiplier = min(2.0, 1 + recent_volatility * 10)

        # Adaptive thresholds based on market conditions
        base_symmetry = self.config.get('min_cup_symmetry', 0.25)
        base_roundness = self.config.get('min_cup_roundness', 0.35)
        
        # STEP 3: Log pattern details (using complete_patterns)
        for i, p in enumerate(complete_patterns[:5]):  # Show first 5 for debugging
            score = p.get('quality_score', 0)
            passes = score >= self.config['min_quality_score']
            symmetry = p.get('cup_symmetry', 0)
            roundness = p.get('cup_roundness', 0)
            
            # Safe duration calculation (we know timestamps are valid now)
            cup_duration_hours = p.get('cup_duration_min', 0) / 60
            total_duration_hours = ((pd.to_datetime(p['breakout_e']) - pd.to_datetime(p['peak_a'])).total_seconds() / 3600)
            
            print(f"  Pattern {i+1}: score={score:.2f}, passes={passes}")
            print(f"    Cup duration: {cup_duration_hours:.1f}h, Total: {total_duration_hours:.1f}h")
            print(f"    Cup depth: {p.get('cup_depth_pct', 0):.1f}%, Handle: {p.get('handle_depth_pct', 0):.1f}%")
            print(f"    Symmetry={symmetry:.2f}, Roundness={roundness:.2f}")
        
        # STEP 4: Apply quality filters to complete patterns
        filtered = []
        for p in complete_patterns:  # ← Use complete_patterns, not original patterns
            # Basic quality score
            if p.get('quality_score', 0) < self.config['min_quality_score']:
                continue

            timeframe_minutes = self.detect_timeframe(df) if hasattr(self, 'detect_timeframe') else 15
            cup_duration_hours = p.get('cup_duration_min', 0) / 60
            
            if timeframe_minutes <= 15:  # Intraday timeframes
                max_cup_hours = 72  # Maximum 12 hours for 15min patterns
                if cup_duration_hours > max_cup_hours:
                    print(f"   ❌ FILTERED: Cup too long for {timeframe_minutes}min timeframe: {cup_duration_hours:.1f}h > {max_cup_hours}h")
                    continue

            if hasattr(self, '_processed_df') and self._processed_df is not None:
                current_atr = self.calculate_atr(self._processed_df, 20).iloc[-1]
                avg_price = self._processed_df['close'].tail(20).mean()
                min_cup_depth_pct = (current_atr * 2.0 / avg_price) * 100  # 2x ATR as percentage
            else:
                min_cup_depth_pct = 0.3 # Default to 0.5% if no processed_df available
            
            if p.get('cup_depth_pct', 0) < min_cup_depth_pct:
                print(f"   ❌ FILTERED: Cup too shallow {p.get('cup_depth_pct', 0):.1f}% < {min_cup_depth_pct:.1f}%")
                continue

            min_cup_symmetry = base_symmetry / volatility_multiplier 
            if p.get('cup_symmetry', 0) < min_cup_symmetry:
                print(f"   ❌ FILTERED: Cup too asymmetrical {p.get('cup_symmetry', 0):.2f} < {min_cup_symmetry}")
                continue
                
            min_cup_roundness = base_roundness / volatility_multiplier  
            if p.get('cup_roundness', 0) < min_cup_roundness:
                print(f"   ❌ FILTERED: Cup too V-shaped {p.get('cup_roundness', 0):.2f}")
                continue
                
            filtered.append(p)

        logger.info(f"Filtered from {len(patterns)} to {len(filtered)} high-quality patterns")
        return filtered
    
    def validate_handle_depth_adaptive(self, timeframe_minutes, peak_c_price, handle_d_price):
        """Adaptive handle depth validation based on timeframe"""
        
        handle_drop_points = peak_c_price - handle_d_price
        handle_drop_percent = (handle_drop_points / peak_c_price) * 100
        
        base_min_drop = self.config.get('min_handle_drop', 0.8)
        base_max_drop = self.config.get('max_handle_drop', 25.0)

        if timeframe_minutes <= 15:      # Intraday (1-15min)
            min_drop_percent = base_min_drop * 0.6   # Scale down for intraday
            max_drop_percent = base_max_drop * 0.6
            min_drop_points = 0.25
        elif timeframe_minutes <= 60:    # Short-term (30-60min)
            min_drop_percent = base_min_drop * 0.8
            max_drop_percent = base_max_drop * 0.8
            min_drop_points = 0.50
        elif timeframe_minutes <= 240:   # 4H timeframe
            min_drop_percent = base_min_drop
            max_drop_percent = base_max_drop
            min_drop_points = 0.75
        else:                           # Daily+
            min_drop_percent = base_min_drop * 1.5   # Scale up for daily
            max_drop_percent = base_max_drop * 1.3
            min_drop_points = 1.0
        
        # Must meet BOTH percentage AND absolute criteria
        if handle_drop_percent < min_drop_percent or handle_drop_points < min_drop_points:
            return False, f"handle_too_shallow_{handle_drop_percent:.2f}pct"
        if handle_drop_percent > max_drop_percent:
            return False, f"handle_too_deep_{handle_drop_percent:.2f}pct"
            
        return True, f"valid_handle_depth_{handle_drop_percent:.2f}pct"
   

    
    
    def get_adaptive_parameters(self, df, timeframe_minutes):
        """Get parameters adapted to current market conditions and timeframe"""
        
        # Calculate recent market volatility
        recent_data = df.tail(100)  # Last 100 bars
        avg_true_range = self.calculate_atr(recent_data, 14)
        volatility_factor = avg_true_range / recent_data['close'].mean()
        
        # Base parameters by timeframe
        if timeframe_minutes <= 15:      # Intraday
            base_config = {
                "min_cup_depth": 0.05,      # 0.3%
                "max_cup_depth": 8.0,       # 8%
                "min_cup_duration": 60,      # 1 hour
                "max_cup_duration": 1440,    # 1 day
                "rim_tolerance_base": 0.5,   # 0.5%
            }
        elif timeframe_minutes <= 60:    # Short-term
            base_config = {
                "min_cup_depth": 0.005,      # 0.5%
                "max_cup_depth": 0.12,       # 12%
                "min_cup_duration": 180,     # 3 hours
                "max_cup_duration": 4320,    # 3 days
                "rim_tolerance_base": 1.0,   # 1%
            }
        else:                           # Daily+
            base_config = {
                "min_cup_depth": 0.08,       # 8%
                "max_cup_depth": 0.5,        # 50%
                "min_cup_duration": 3600,    # 2.5 days
                "max_cup_duration": 129600,  # 90 days
                "rim_tolerance_base": 2.0,   # 2%
            }
        
        # ADJUST based on volatility
        volatility_multiplier = 1 + (volatility_factor * 2)  # More volatile = more lenient
        
        adjusted_config = base_config.copy()
        adjusted_config["min_cup_depth"] *= (0.5 + volatility_factor)  # Shallower cups in volatile markets
        adjusted_config["rim_tolerance_base"] *= volatility_multiplier  # More tolerance in volatile markets
        
        return adjusted_config
    

    def find_best_left_rim(self, df, accumulation_start, resistance_level, max_search_bars=50):
        """Find actual left rim peak, not just accumulation start"""
        
        try:
            acc_start_idx = df.index.get_loc(accumulation_start)
            search_start = max(0, acc_start_idx - max_search_bars)
            search_end = acc_start_idx + 5  # Small buffer after accumulation start
            
            # Look for actual peaks in this window
            search_window = df.iloc[search_start:search_end]
            peaks = search_window[search_window['extrema'] == 1]
            
            print(f"    🔍 Left rim search: {len(peaks)} peaks found near accumulation start")
            
            if len(peaks) == 0:
                print(f"    ⚠️ No peaks found, using accumulation start")
                return accumulation_start, df.loc[accumulation_start, 'high']
            
            # Find peak closest to resistance level
            best_peak = None
            min_diff = float('inf')
            
            for peak_time, peak_row in peaks.iterrows():
                peak_price = peak_row['high']
                diff = abs(peak_price - resistance_level)
                
                # Also check if peak is reasonably close to resistance (within 5%)
                if diff / resistance_level <= 0.05:
                    if diff < min_diff:
                        min_diff = diff
                        best_peak = (peak_time, peak_price)
                        print(f"    🎯 Better left rim candidate: {peak_time} at ${peak_price:.2f} (diff: ${diff:.2f})")
            
            if best_peak:
                print(f"    ✅ Selected left rim: {best_peak[0]} at ${best_peak[1]:.2f}")
                return best_peak[0], best_peak[1]
            
            print(f"    ⚠️ No suitable peaks found, using accumulation start")
            return accumulation_start, df.loc[accumulation_start, 'high']
            
        except Exception as e:
            print(f"    ❌ Left rim search failed: {e}, using accumulation start")
            return accumulation_start, df.loc[accumulation_start, 'high']
 


    def detect_actual_cup_formations(self, df, resistance_levels):
            """
            REPLACE your detect_accumulation_base() with this method.
            Finds actual cup shapes, not random sideways movement.
            """
            
            valid_cups = []
            timeframe_minutes = self.detect_timeframe(df)
            
            min_cup_minutes = self.config.get('min_cup_duration', 2880)
            max_cup_minutes = self.config.get('max_cup_duration', 14400)
            min_cup_bars = max(10, min_cup_minutes // timeframe_minutes)
            max_cup_bars = max(20, max_cup_minutes // timeframe_minutes)
            
            print(f"🔍 ACTUAL CUP DETECTION: Looking for {min_cup_bars}-{max_cup_bars} bar formations")
            
            # For each resistance level, look for cup formations that reach it
            for resistance in resistance_levels[:20]:  # Top 20 resistance levels
                resistance_price = resistance['price']
                print(f"\n   📊 Checking resistance ${resistance_price:.2f}")
                
                # Scan through data looking for cup formations near this resistance
                for start_idx in range(len(df) - min_cup_bars):
                    for cup_length in range(min_cup_bars, min(max_cup_bars, len(df) - start_idx)):
                        end_idx = start_idx + cup_length
                        cup_segment = df.iloc[start_idx:end_idx]
                        
                        # 1. CHECK: Do the rims reach our resistance level?
                        left_rim_price = cup_segment['high'].iloc[:3].max()   # First 3 bars max
                        right_rim_price = cup_segment['high'].iloc[-3:].max() # Last 3 bars max
                        
                        # Both rims must be close to resistance level
                        left_diff = abs(left_rim_price - resistance_price) / resistance_price
                        right_diff = abs(right_rim_price - resistance_price) / resistance_price
                        
                        if left_diff > 0.05 or right_diff > 0.05:  # More than 5% away from resistance
                            continue
                        
                        # 2. VALIDATE: Actual cup shape exists
                        is_valid_shape, shape_reason = self.validate_actual_cup_shape(df, start_idx, end_idx)
                        if not is_valid_shape:
                            continue
                        
                        # 3. CHECK: Rim symmetry (within 8% for real markets)
                        rim_diff_pct = abs(left_rim_price - right_rim_price) / max(left_rim_price, right_rim_price) * 100
                        if rim_diff_pct > 8.0:
                            continue
                        
                        # 4. CHECK: Meaningful cup depth
                        cup_bottom = cup_segment['low'].min()
                        cup_depth_pct = ((max(left_rim_price, right_rim_price) - cup_bottom) / 
                                    max(left_rim_price, right_rim_price) * 100)
                        if cup_depth_pct < 5.0:  # Minimum 5% depth
                            continue
                        
                        # 5. FOUND: Valid cup formation
                        valid_cups.append({
                            'cup_start': df.index[start_idx],
                            'cup_end': df.index[end_idx],
                            'left_rim_price': left_rim_price,
                            'right_rim_price': right_rim_price,
                            'cup_bottom': cup_bottom,
                            'cup_depth_pct': cup_depth_pct,
                            'rim_symmetry_pct': rim_diff_pct,
                            'resistance_level': resistance_price,
                            'shape_validation': shape_reason,
                            'cup_bars': cup_length
                        })
                        
                        print(f"      ✅ FOUND CUP: {df.index[start_idx]} to {df.index[end_idx]}")
                        print(f"         Depth: {cup_depth_pct:.1f}%, Rim diff: {rim_diff_pct:.1f}%")
                        print(f"         Shape: {shape_reason}")
                        
                        # Only find one cup per resistance level to avoid duplicates
                        break
                    else:
                        continue
                    break
            
            print(f"\n📊 CUP DETECTION RESULTS: {len(valid_cups)} actual cup formations found")
            return valid_cups
    

    def validate_actual_cup_shape(self, df, start_idx, end_idx, min_depth_pct=3.0):
        """
        Professional geometric validation - ensures actual cup shape exists.
        This replaces parameter tuning with fundamental shape requirements.
        """
        
        if end_idx - start_idx < 15:  # Need minimum bars for cup analysis
            return False, "Insufficient data for cup analysis"
        
        segment = df.iloc[start_idx:end_idx]
        
        # 1. DESCENT PHASE VALIDATION (First 30-60% of pattern)
        total_bars = len(segment)
        descent_end = total_bars // 3  # First third
        descent_phase = segment.iloc[:descent_end]
        
        if len(descent_phase) < 3:
            return False, "Descent phase too short"
        
        # Must have meaningful descent (not sideways drift)
        start_price = descent_phase['close'].iloc[0]
        descent_low = descent_phase['low'].min()
        descent_depth = (start_price - descent_low) / start_price * 100
        
        if descent_depth < min_depth_pct:
            return False, f"No meaningful descent: {descent_depth:.1f}% < {min_depth_pct}%"
        
        # 2. RECOVERY PHASE VALIDATION (Last 30-60% of pattern)  
        recovery_start = (total_bars * 2) // 3  # Last third
        recovery_phase = segment.iloc[recovery_start:]
        
        if len(recovery_phase) < 3:
            return False, "Recovery phase too short"
        
        # Must have meaningful recovery
        recovery_start_price = recovery_phase['close'].iloc[0]
        recovery_end_price = recovery_phase['close'].iloc[-1]
        recovery_gain = (recovery_end_price - recovery_start_price) / recovery_start_price * 100
        
        if recovery_gain < min_depth_pct * 0.7:  # At least 70% of descent depth
            return False, f"Insufficient recovery: {recovery_gain:.1f}% < {min_depth_pct*0.7:.1f}%"
        
        # 3. BOTTOM ACCUMULATION VALIDATION (Prevents V-spike reversals)
        cup_low = segment['low'].min()
        cup_high = segment['high'].max()
        cup_range = cup_high - cup_low
        
        # Define bottom zone (lowest 25% of total range)
        bottom_threshold = cup_low + (cup_range * 0.25)
        
        # Count bars spending time in bottom zone
        bottom_bars = len(segment[segment['low'] <= bottom_threshold])
        bottom_time_pct = bottom_bars / total_bars * 100
        
        # Real cups spend significant time building a base
        if bottom_time_pct < 20.0:  # Less than 20% time at bottom
            return False, f"V-spike reversal: only {bottom_time_pct:.1f}% time at bottom"
        
        # 4. VOLATILITY CONSISTENCY (No wild swings during cup formation)
        daily_ranges = (segment['high'] - segment['low']) / segment['close'] * 100
        avg_daily_range = daily_ranges.mean()
        max_daily_range = daily_ranges.max()
        
        # Maximum single-day range shouldn't exceed 3x average
        if max_daily_range > avg_daily_range * 3:
            return False, f"Excessive volatility: {max_daily_range:.1f}% vs avg {avg_daily_range:.1f}%"
        
        # 5. MONOTONIC BIAS CHECK (Price shouldn't trend strongly up/down)
        # Calculate linear trend over cup period
        x = np.arange(len(segment))
        y = segment['close'].values
        
        # Fit linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate trend as percentage of starting price
        trend_pct = (slope * len(segment)) / y[0] * 100
        
        # Strong trends indicate this isn't a consolidation cup
        if abs(trend_pct) > 8.0:  # More than 8% trend
            return False, f"Strong trend detected: {trend_pct:.1f}% - not a cup"
        
        return True, f"Valid cup: descent={descent_depth:.1f}%, recovery={recovery_gain:.1f}%, bottom_time={bottom_time_pct:.1f}%"

    def find_handle_after_cup(self, df, cup_end_idx, right_rim_price):
        """Find handle formation after cup end"""
        
        if cup_end_idx >= len(df) - 5:
            return None
        
        # Look for pullback in next 20 bars
        timeframe_minutes = self.detect_timeframe(df) if hasattr(self, 'detect_timeframe') else 15
        handle_search_minutes = self.config.get('handle_search_duration', 960)  # Default 16 hours  
        handle_search_bars = max(20, handle_search_minutes // timeframe_minutes)
        search_window = df.iloc[cup_end_idx:cup_end_idx + handle_search_bars]

        min_handle_duration_minutes = self.config.get('min_handle_duration', 240)  # Default from config
        min_handle_bars = max(2, min_handle_duration_minutes // timeframe_minutes)

        for i in range(min_handle_bars, len(search_window)):
            handle_segment = search_window.iloc[:i]
            handle_low = handle_segment['low'].min()
            
            # Handle depth check
            handle_depth_pct = (right_rim_price - handle_low) / right_rim_price * 100
            
            if 1.0 <= handle_depth_pct <= 25.0:  # Valid handle depth
                return {
                    'start': handle_segment.index[0],
                    'end': handle_segment.index[-1],
                    'depth_pct': handle_depth_pct,
                    'low_price': handle_low
                }
        print(f"❌ PATTERN REJECTED AT FINAL STAGE: [add reason here]")
        return None

    
    def detect(self, df: pd.DataFrame, extrema_col="extrema", price_col="close", return_rejections=False):
        """Main detection function that runs the full pipeline."""
        logger.info("Starting Cup and Handle pattern detection")
        print(f"🔧 CONFIG AFTER UPDATE:")
        print(f"   rim_height_tolerance_pct: {self.config.get('rim_height_tolerance_pct')}")
        print(f"   min_handle_drop: {self.config.get('min_handle_drop')}")
        print(f"   min_cup_depth: {self.config.get('min_cup_depth')}")
        print(f"   min_cup_duration: {self.config.get('min_cup_duration')}")
        # Step 1: Preprocess data
        logger.info("Preprocessing data")
        processed_df = self.preprocess_data(df, price_col)
        self._processed_df = processed_df

        print(f"DEBUG INDEX CHECK - Index type: {type(processed_df.index)}")
        print(f"DEBUG INDEX CHECK - First index: {processed_df.index[0]}")
        print(f"DEBUG INDEX CHECK - Index name: {processed_df.index.name}")
        
        # Step 2: Detect multi-scale extrema
        logger.info("Detecting extrema")
        processed_df['extrema'] = self.detect_extrema_multi_scale(processed_df, f"{price_col}_smooth")
        extrema_points = processed_df[processed_df['extrema'] != 0]
        peaks = extrema_points[extrema_points['extrema'] == 1]
        troughs = extrema_points[extrema_points['extrema'] == -1]
        print(f"DEBUG EXTREMA - Total: {len(extrema_points)}, Peaks: {len(peaks)}, Troughs: {len(troughs)}")
        print(f"DEBUG EXTREMA - First 10 peaks: {peaks.index[:10].tolist()}")
        print(f"DEBUG EXTREMA - First 10 troughs: {troughs.index[:10].tolist()}")
        print(f"🔍 Total peaks: {len(peaks)}, troughs: {len(troughs)}")
        logger.info("Detecting extrema")
        processed_df['extrema'] = self.detect_extrema_multi_scale(processed_df, f"{price_col}_smooth")
        extrema_points = processed_df[processed_df['extrema'] != 0]
        peaks = extrema_points[extrema_points['extrema'] == 1]
        troughs = extrema_points[extrema_points['extrema'] == -1]
        print(f"🔍 Total peaks: {len(peaks)}, troughs: {len(troughs)}")

        # ADD THIS TEST CODE HERE:
        print("="*50)
        print("TESTING FORMATION-FIRST DETECTION")
        print("="*50)

   
   
        print(f"🔍 EXTREMA DETECTION RESULTS:")
        print(f"   Total extrema points: {len(extrema_points)}")
        print(f"   Peaks: {len(peaks)}")
        print(f"   Troughs: {len(troughs)}")
        print(f"   First 10 extrema indices: {extrema_points.index[:10].tolist()}")
        print("\n" + "="*60)
        print("🔍 TESTING FORMATION-FIRST DETECTION")
        print("="*60)


        # Step 3: Detect Cup and Handle patterns
        logger.info("Detecting Cup and Handle patterns")
        

        # Store rejected pattern info for manual review
     
        patterns, rejection_stats = self.detect_cup_and_handle(processed_df, 'extrema', f"{price_col}_smooth")

        # Add simple formation-first patterns if enabled
        if self.config.get('enable_formation_first', False):
            patterns.extend(self._add_formation_patterns(processed_df))

        patterns = self.remove_duplicate_patterns(patterns)
        print(f"\n🚨 PATTERN FLOW DEBUG:")
        print(f"   Patterns returned from detect_cup_and_handle: {len(patterns)}")
        if patterns:
            first_pattern = patterns[0]
            print(f"   First pattern keys: {list(patterns[0].keys())}")
            print(f"   First pattern duration: {patterns[0].get('cup_duration_min', 0)/60:.1f}h")
        else:
            print(f"   ❌ No patterns returned from institutional detection!")
        # Step 4: Filter high-quality patterns
        logger.info("Filtering high-quality patterns")
        filtered_patterns = self.filter_patterns(patterns)
        logger.info("Fine-tuning patterns for optimal rim heights")
     
        if self.config.get('skip_rim_adjustment', False):
         logger.info(f"Skipping rim adjustment - using original {len(filtered_patterns)} patterns")

    
   
        # Return both original and filtered patterns
        if return_rejections:
           return filtered_patterns, rejection_stats



        # Generate detailed analysis report with error handling
        try:
            # self.generate_detailed_analysis_report(processed_df)

            if hasattr(self, 'deep_cup_log') and self.deep_cup_log:
                with open('deep_cup_analysis.log', 'w') as f:
                    f.write("DEEP CUP ANALYSIS LOG\n")
                    f.write("=" * 50 + "\n\n")
                    for entry in self.deep_cup_log:
                        f.write(entry + "\n")
                print(f"📄 Deep cup analysis saved to: deep_cup_analysis.log")
        except Exception as e:
            print(f"⚠️ Analysis report failed: {e}")
            print("Continuing with pattern detection results...")

        print("\n" + "="*60)

        print("="*60)


        return filtered_patterns
    
    def detect_combined(self, df: pd.DataFrame, extrema_col="extrema", price_col="close", return_rejections=False):
        """Main detection function that runs both strict and relaxed versions in parallel."""
        logger.info("Starting Combined Cup and Handle pattern detection")
        
        # Step 1: Shared preprocessing (memory efficient)
        logger.info("Preprocessing data (shared)")
        processed_df = self.preprocess_data(df, price_col)
        self._processed_df = processed_df
        
        # Step 2: Shared extrema detection (memory efficient)
        logger.info("Detecting extrema (shared)")
        processed_df['extrema'] = self.detect_extrema_multi_scale(processed_df, f"{price_col}_smooth")
        
        # Step 3: Run both detection versions
        logger.info("Running strict version")
        strict_config = self.get_strict_config()
        strict_patterns = self._run_detection_with_config(processed_df, strict_config, 'strict', extrema_col, price_col)
        
        logger.info("Running relaxed version") 
        relaxed_config = self.get_relaxed_config()
        relaxed_patterns = self._run_detection_with_config(processed_df, relaxed_config, 'relaxed', extrema_col, price_col)
        
        # Step 4: Combine results with source tags
        combined_patterns = self.merge_pattern_results(strict_patterns, relaxed_patterns)
        
        return combined_patterns

    def get_strict_config(self):
        """Strict config for higher-quality, longer-term patterns"""
        return {
            "min_cup_roundness": 0.4,
            "cup_symmetry_threshold": 0.2,
            "prominence": 0.001,
            "rim_height_tolerance_pct": 1.0,
            "min_quality_score": 65,           # Higher quality threshold
            "min_cup_depth": 0.01,            # Deeper cups (1%+)
            "min_cup_duration": 720,          # Longer cups (12+ hours)
            "max_handle_depth_pct": 35.0      # Shallower handles
        }

    def get_relaxed_config(self):
        """Relaxed config for catching more patterns"""
        return {
            "min_cup_roundness": 0.2,
            "cup_symmetry_threshold": 0.15,
            "prominence": 0.002,
            "rim_height_tolerance_pct": 2.0,
            "min_quality_score": 40,           # Lower threshold
            "min_cup_depth": 0.005,           # Allow shallower
            "min_cup_duration": 360,          # Shorter duration ok
            "max_handle_depth_pct": 45.0      # Allow deeper handles
        }
    
    def _run_detection_with_config(self, processed_df, config_override, source_tag, extrema_col='extrema', price_col='close_smooth'):
        """Run detection with specific config and tag results with source."""
        
        # Backup original config
        original_config = self.config.copy()
        
        try:
            # Temporarily update config
            self.config.update(config_override)
            
            # Run detection with modified config
            patterns, rejection_stats = self.detect_cup_and_handle(processed_df, extrema_col, price_col)
            
            # Remove duplicates
            patterns = self.remove_duplicate_patterns(patterns)
            
            # Filter patterns
            filtered_patterns = self.filter_patterns(patterns)
            
            # Tag each pattern with source
            for pattern in filtered_patterns:
                pattern['detection_source'] = source_tag
                pattern['config_used'] = config_override.copy()
            
            logger.info(f"{source_tag} detection found {len(filtered_patterns)} patterns")
            return filtered_patterns
            
        finally:
            # Always restore original config
            self.config = original_config

    def merge_pattern_results(self, strict_patterns, relaxed_patterns):
        """Merge results from both detection methods with overlap handling."""
        
        combined_results = {
            'strict_patterns': strict_patterns,
            'relaxed_patterns': relaxed_patterns,
            'all_patterns': [],
            'overlap_analysis': {
                'strict_only': 0,
                'relaxed_only': 0,
                'overlapping': 0,
                'total_unique_periods': 0
            }
        }
        
        # Simple merge: add all patterns with source tags
        all_patterns = []
        
        # Add strict patterns
        for pattern in strict_patterns:
            pattern['pattern_id'] = f"strict_{len(all_patterns)}"
            all_patterns.append(pattern)
        
        # Add relaxed patterns
        for pattern in relaxed_patterns:
            pattern['pattern_id'] = f"relaxed_{len(all_patterns)}"
            all_patterns.append(pattern)
        
        # Sort by start time for easier analysis
        all_patterns.sort(key=lambda x: x['peak_a'])
        
        # Basic overlap detection (for statistics)
        strict_periods = [(p['peak_a'], p['breakout_e']) for p in strict_patterns]
        relaxed_periods = [(p['peak_a'], p['breakout_e']) for p in relaxed_patterns]
        
        # Count overlaps
        overlaps = 0
        for s_start, s_end in strict_periods:
            for r_start, r_end in relaxed_periods:
                # Check if periods overlap
                if (s_start <= r_end and s_end >= r_start):
                    overlaps += 1
                    break
        
        # Update statistics
        combined_results['overlap_analysis'] = {
            'strict_only': len(strict_patterns) - overlaps,
            'relaxed_only': len(relaxed_patterns) - overlaps, 
            'overlapping': overlaps,
            'total_patterns': len(all_patterns),
            'unique_periods': len(strict_patterns) + len(relaxed_patterns) - overlaps
        }
        
        combined_results['all_patterns'] = all_patterns
        
        logger.info(f"Combined results: {len(strict_patterns)} strict + {len(relaxed_patterns)} relaxed = {len(all_patterns)} total patterns")
        logger.info(f"Overlap analysis: {overlaps} overlapping periods detected")
        
        return combined_results
        

    def _add_formation_patterns(self, df):
        """Simple formation-first detection - just finds A-B-C-D-E sequences"""
        patterns = []
        peaks = df[df['extrema'] == 1]
        troughs = df[df['extrema'] == -1]
        
        # Process only first 20 peaks to keep it simple
        for i, peak_a in enumerate(peaks.index[:20]):
            if i >= len(peaks) - 2: break
            
            peak_c_candidates = peaks[peaks.index > peak_a][:5]  # Next 5 peaks only
            
            for peak_c in peak_c_candidates.index:
                # Find trough between them
                between_troughs = troughs[(troughs.index > peak_a) & (troughs.index < peak_c)]
                if len(between_troughs) == 0: continue
                
                trough_b = between_troughs.index[0]  # First trough
                
                # Quick validation using your existing methods
                try:
                    pattern = self.create_institutional_pattern(
                        {'price': df.loc[peak_c, 'high'], 'touches': 2},
                        {'start': peak_a, 'end': peak_c, 'score': 50},
                        {'start': peak_c, 'end': peak_c, 'depth_pct': 1.0, 'score': 50, 'low_price': df.loc[peak_c, 'low']},
                        peak_c, df, peak_c, 'close_smooth'
                    )
                    if pattern:
                        pattern['detection_method'] = 'formation_first'
                        patterns.append(pattern)
                        if len(patterns) >= 10: break  # Limit to 10 additional patterns
                except:
                    continue
        
        print(f"   📊 Formation-first added: {len(patterns)} patterns")
        return patterns

   
   

    def validate_minimum_cup_duration(self, cup_start, cup_end, timeframe_minutes=15):
        """Ensure cup spans sufficient time for institutional relevance."""
        cup_duration_minutes = (cup_end - cup_start).total_seconds() / 60
        minimum_hours = 8  # Minimum 8 hours for a valid cup
        
        if cup_duration_minutes < minimum_hours * 60:
            return False, f"Cup too short: {cup_duration_minutes/60:.1f}h < {minimum_hours}h"
        
        # Also check number of bars
        expected_bars = cup_duration_minutes / timeframe_minutes
        if expected_bars < 20:  # Minimum 20 bars
            return False, f"Cup too few bars: {expected_bars:.0f} < 20"
        
        return True, f"Duration valid: {cup_duration_minutes/60:.1f}h"

      
    
    def detect_timeframe(self, df):
        """Detect the timeframe of the data by analyzing time differences."""
        if len(df) < 2:
            return 15  # Default fallback
        
        # Calculate time differences between consecutive bars
        time_diffs = pd.Series(df.index).diff().dropna()
        
        # Get the most common time difference (mode)
        most_common_diff = time_diffs.mode()
        
        if len(most_common_diff) > 0:
            # Convert to minutes
            timeframe_minutes = most_common_diff.iloc[0].total_seconds() / 60
            return int(timeframe_minutes)
        else:
            return 15  # Default fallback

    
    # Improvements to your existing visualize_pattern method in CupHandleDetector class

    def visualize_pattern(self, pattern, df, save_path=None):
        """Generate visualization of a detected pattern with proper traditional candlesticks."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        
        # Extract key timestamps
        timestamps = {
            "peak_a": pattern["peak_a"],
            "trough_b": pattern["trough_b"],
            "peak_c": pattern["peak_c"],
            "handle_d": pattern["handle_d"],
            "breakout_e": pattern["breakout_e"]
        }
        
        # Create window with padding
        pattern_start = pd.to_datetime(timestamps["peak_a"])
        pattern_end = pd.to_datetime(timestamps["breakout_e"])
        start_time = pattern_start - pd.Timedelta(minutes=15)
        end_time = pattern_end + pd.Timedelta(minutes=15)
        
        # Filter data to the window
        window_df = df.loc[start_time:end_time].copy()
        
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]}, 
                                    sharex=True)
        
        # Set the x-axis limits
        ax1.set_xlim(start_time, end_time)
        
        # IMPROVED TRADITIONAL CANDLESTICK RENDERING
        for idx, row in window_df.iterrows():
            # Only plot if we have all required OHLC data
            if 'open' in row and 'high' in row and 'low' in row and 'close' in row:
                if pd.notna(row['open']) and pd.notna(row['high']) and pd.notna(row['low']) and pd.notna(row['close']):
                    # Plot the wick (high to low)
                    ax1.plot([idx, idx], [row['low'], row['high']], color='black', linewidth=1, zorder=1)
                    
                    # Determine candle color
                    if row['close'] >= row['open']:
                        color = 'green'
                        body_bottom = row['open']
                        body_height = row['close'] - row['open']
                    else:
                        color = 'red'
                        body_bottom = row['close']
                        body_height = row['open'] - row['close']
                    
                    # Ensure minimum body height for visibility
                    body_height = max(body_height, 0.01)
                    
                    # Plot candle body
                    body_width = pd.Timedelta(minutes=0.6)  # Width of candle body
                    rect = Rectangle(
                        (idx - body_width/2, body_bottom),
                        body_width,
                        body_height,
                        facecolor=color,
                        edgecolor='black',
                        linewidth=0.5,
                        zorder=2
                    )
                    ax1.add_patch(rect)
        
        # Get prices at key pattern points
        prices = {}
        for point in ["peak_a", "trough_b", "peak_c", "handle_d", "breakout_e"]:
            ts = pd.to_datetime(timestamps[point])
            
            # Find appropriate price level based on point type
            try:
                if point in ["peak_a", "peak_c"]:
                    closest_ts = window_df.index[window_df.index.get_indexer([ts], method='nearest')[0]]
                    prices[point] = window_df.loc[closest_ts, "high"]
                elif point in ["trough_b", "handle_d"]:
                    closest_ts = window_df.index[window_df.index.get_indexer([ts], method='nearest')[0]]
                    prices[point] = window_df.loc[closest_ts, "low"]
                else:  # breakout
                    closest_ts = window_df.index[window_df.index.get_indexer([ts], method='nearest')[0]]
                    prices[point] = window_df.loc[closest_ts, "high"]  # ← Use high for breakout
            except:
                # Fallback for any missing data
                closest_idx = min(range(len(window_df)), key=lambda i: abs(window_df.index[i] - ts))
                prices[point] = window_df.iloc[closest_idx]["close"]
        
        # Define point colors and labels
        point_colors = {
            "peak_a": "red",
            "trough_b": "blue",
            "peak_c": "red",
            "handle_d": "blue",
            "breakout_e": "green"
        }
        
        point_labels = {
            "peak_a": "A: Left Cup Rim",
            "trough_b": "B: Cup Bottom",
            "peak_c": "C: Right Cup Rim",
            "handle_d": "D: Handle",
            "breakout_e": "E: Breakout"
        }
        
        # Plot the cup and handle pattern line
        cup_handle_x = [pd.to_datetime(timestamps[p]) for p in ["peak_a", "trough_b", "peak_c", "handle_d", "breakout_e"]]
        cup_handle_y = []
        
        for p in ["peak_a", "trough_b", "peak_c", "handle_d", "breakout_e"]:
            if p == "peak_a" and "adjusted_a_price" in pattern:
                cup_handle_y.append(pattern["adjusted_a_price"])
            elif p == "peak_c" and "adjusted_c_price" in pattern:
                cup_handle_y.append(pattern["adjusted_c_price"])
            else:
                cup_handle_y.append(prices[p])

        # Plot each segment separately to handle potential gaps
        for i in range(len(cup_handle_x) - 1):
            # Get segment endpoints
            start_x, end_x = cup_handle_x[i], cup_handle_x[i+1]
            start_y, end_y = cup_handle_y[i], cup_handle_y[i+1]
            
            # Create a time index between start and end for this segment
            segment_df = window_df.loc[start_x:end_x].copy()
            
            if len(segment_df) > 1:
                # Check for large time gaps (non-trading periods)
                time_diffs = np.diff(segment_df.index)
                max_normal_diff = pd.Timedelta(minutes=30)  # Expected time between bars
                
                if np.any(time_diffs > max_normal_diff * 3):  # If gaps 3x larger than normal
                    # There's a gap - plot with dashed line to indicate discontinuity
                    ax1.plot([start_x, end_x], [start_y, end_y], color='orange', 
                            linestyle='--', linewidth=2, alpha=0.8)
                else:
                    # No major gaps - plot solid line
                    ax1.plot([start_x, end_x], [start_y, end_y], color='orange',
                            linewidth=2, alpha=1.0)
            else:
                # Not enough points, just draw a line
                ax1.plot([start_x, end_x], [start_y, end_y], color='orange',
                        linewidth=2, alpha=1.0)

        price_range = window_df['high'].max() - window_df['low'].min()
        offset = price_range * 0.02  # 2% vertical offset

        for i, point in enumerate(["peak_a", "trough_b", "peak_c", "handle_d", "breakout_e"]):
            y = cup_handle_y[i]
            if point in ["peak_a", "peak_c", "breakout_e"]:
                y += offset  # move above
            else:
                y -= offset  # move below

            ax1.scatter(
                cup_handle_x[i], 
                y, 
                color=point_colors[point], 
                s=80, 
                zorder=5
            )
        
        # Add resistance line
        resistance_level = max(prices["peak_a"], prices["peak_c"])
        ax1.axhline(y=resistance_level, color='red', linestyle='--', 
                linewidth=1, alpha=0.7, label="Resistance Level")
        
        # # Add annotations with better styling
        # for point in ["peak_a", "trough_b", "peak_c", "handle_d", "breakout_e"]:
        #     # Create box properties
        #     bbox_props = dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8, edgecolor="black")
            
        #     # Determine vertical position of label
        #     price_range = window_df['high'].max() - window_df['low'].min()
        #     dynamic_offset = price_range * 0.02  # 2% of price range

        #     # Determine vertical position of label
        #     if point in ["peak_a", "peak_c", "breakout_e"]:
        #         y_offset = dynamic_offset  # Above the point
        #         va = 'bottom'
        #     else:
        #         y_offset = -dynamic_offset  # Below the point
        #         va = 'top'
            
            # # Add the annotation with arrow
            # ax1.annotate(
            #     point_labels[point],
            #     xy=(pd.to_datetime(timestamps[point]), prices[point]),
            #     xytext=(pd.to_datetime(timestamps[point]), prices[point] + y_offset),
            #     color='black',
            #     fontsize=9,
            #     ha='center',
            #     va=va,
            #     bbox=bbox_props,
            #     arrowprops=dict(arrowstyle="->", color="black", lw=1.5)
            # )
        
        # Calculate pattern metrics
        cup_depth = (resistance_level - prices["trough_b"]) / resistance_level * 100
        cup_duration = (pd.to_datetime(timestamps["peak_c"]) - pd.to_datetime(timestamps["peak_a"])).total_seconds() / 60
        handle_depth = (prices["peak_c"] - prices["handle_d"]) / prices["peak_c"] * 100
        handle_duration = (pd.to_datetime(timestamps["breakout_e"]) - pd.to_datetime(timestamps["handle_d"])).total_seconds() / 60
        
        # Get quality and confidence scores
        quality_score = pattern.get('quality_score', 0)
        confidence = pattern.get('confidence_score', 0)
        
        # Add metrics to title
        date_str = pd.to_datetime(timestamps["peak_a"]).strftime('%Y-%m-%d')
        start_time_str = pd.to_datetime(timestamps["peak_a"]).strftime('%H:%M')
        end_time_str = pd.to_datetime(timestamps["breakout_e"]).strftime('%H:%M')
        
        title = f"Cup and Handle Pattern - {date_str} - {start_time_str} to {end_time_str}\n"
        title += f"Cup Depth: {cup_depth:.2f}% | Cup Duration: {cup_duration:.0f} min | Handle Depth: {handle_depth:.2f}% | Quality: {quality_score:.1f}/100"
        ax1.set_title(title, fontsize=12)
        
      
        # Plot volume or momentum in the bottom panel
        if 'volume' in window_df.columns and window_df['volume'].sum() > 0:
            # Normalize volume for better visibility
            max_volume = window_df['volume'].max()
            if max_volume > 0:  # Avoid division by zero
                normalized_volume = window_df['volume'] / max_volume
                
                # Plot volume bars with same colors as price candles
                for idx, row in window_df.iterrows():
                    if pd.notna(row.get('open')) and pd.notna(row.get('close')):
                        color = 'green' if row['close'] >= row['open'] else 'red'
                        
                        bar_width = pd.Timedelta(minutes=0.6)
                        height = normalized_volume.loc[idx] if idx in normalized_volume.index else 0
                        
                        rect = Rectangle(
                            (idx - bar_width/2, 0),
                            bar_width,
                            height,
                            facecolor=color,
                            alpha=0.7,
                            zorder=2
                        )
                        ax2.add_patch(rect)
                
                ax2.set_ylabel('Volume', fontsize=10)
        else:
            # Use price momentum as fallback
            window_df['momentum'] = window_df['close'].diff(3)
            
            for idx, row in window_df.iterrows():
                if pd.notna(row.get('momentum')):
                    momentum = row['momentum']
                    color = 'green' if momentum >= 0 else 'red'
                    
                    bar_width = pd.Timedelta(minutes=0.6)
                    
                    rect = Rectangle(
                        (idx - bar_width/2, 0 if momentum >= 0 else momentum),
                        bar_width,
                        abs(momentum),
                        facecolor=color,
                        alpha=0.7,
                        zorder=2
                    )
                    ax2.add_patch(rect)
            
            ax2.set_ylabel('Price Momentum', fontsize=10)
        
        # Format axes
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Set y-axis limits with padding
        min_price = window_df['low'].min()
        max_price = window_df['high'].max()
        price_range = max_price - min_price
        padding = price_range * 0.1  # 10% padding
        ax1.set_ylim(min_price - padding, max_price + padding)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved pattern visualization to {save_path}")
        
        return fig
    
    def create_pattern_fingerprint(self, pattern, df, price_col='close_smooth'):
            """Create a detailed geometric fingerprint for pattern analysis."""
            
            # Get pattern points
            A_ts, B_ts, C_ts, D_ts, E_ts = pattern["peak_a"], pattern["trough_b"], pattern["peak_c"], pattern["handle_d"], pattern["breakout_e"]
            A_price = df.loc[A_ts, 'high']    # Peak uses high
            B_price = df.loc[B_ts, 'low']     # Trough uses low  
            C_price = df.loc[C_ts, 'high']    # Peak uses high
            D_price = df.loc[D_ts, 'low']     # Trough uses low
            E_price = df.loc[E_ts, 'high']  
            
            # Basic measurements
            fingerprint = {
                "pattern_id": f"P_{A_ts.strftime('%m%d_%H%M')}",
                
                # Rim analysis
                "left_rim_price": A_price,
                "right_rim_price": C_price,
                "rim_height_diff_pct": abs(A_price - C_price) / max(A_price, C_price) * 100,
                "rim_height_avg": (A_price + C_price) / 2,
                
                # Cup geometry
                "cup_depth_abs": pattern["cup_depth"],
                "cup_depth_pct": pattern["cup_depth_pct"],
                "cup_width_minutes": pattern["cup_duration_min"],
                "cup_symmetry": pattern["cup_symmetry"],
                "cup_roundness": pattern["cup_roundness"],
                
                # Cup shape ratios
                "left_descent_pct": (A_price - B_price) / A_price * 100,
                "right_ascent_pct": (C_price - B_price) / C_price * 100,
                "cup_aspect_ratio": pattern["cup_duration_min"] / (pattern["cup_depth_pct"] * 10),  # width/depth ratio
                
                # Handle geometry
                "handle_depth_abs": pattern["handle_depth"],
                "handle_depth_pct": pattern["handle_depth_pct"],
                "handle_width_minutes": pattern["handle_duration_min"],
                "handle_position": pattern["handle_position"],
                "handle_to_cup_duration_ratio": pattern["handle_to_cup_ratio"],
                
                # Handle shape
                "handle_slope": (D_price - C_price) / pattern["handle_duration_min"] if pattern["handle_duration_min"] > 0 else 0,
                "handle_depth_vs_cup_depth": pattern["handle_depth_pct"] / pattern["cup_depth_pct"],
                
                # Breakout analysis
                "breakout_strength_pct": (E_price - C_price) / C_price * 100,
                "breakout_distance_minutes": (pd.to_datetime(E_ts) - pd.to_datetime(D_ts)).total_seconds() / 60,
                
                # Pattern proportions
                "total_pattern_duration": (pd.to_datetime(E_ts) - pd.to_datetime(A_ts)).total_seconds() / 60,
                "cup_to_total_ratio": pattern["cup_duration_min"] / ((pd.to_datetime(E_ts) - pd.to_datetime(A_ts)).total_seconds() / 60),
                
                # Quality metrics
                "quality_score": pattern["quality_score"],
                "confidence_score": pattern["confidence_score"],
                
                # Visual shape descriptors
                "is_cup_v_shaped": 1 if pattern["cup_roundness"] < 0.3 else 0,
                "is_cup_u_shaped": 1 if pattern["cup_roundness"] > 0.7 else 0,
                "is_handle_steep": 1 if abs(pattern["handle_depth_pct"]) > 15 else 0,
                "is_handle_shallow": 1 if abs(pattern["handle_depth_pct"]) < 5 else 0,
                
                # Timestamps for reference
                "start_time": A_ts,
                "end_time": E_ts,
            }
            
                        
            return fingerprint

    def detect_with_fingerprints(self, df, price_col='close'):
        """Main detection function that includes pattern fingerprinting."""
        logger.info("Starting Cup and Handle pattern detection with fingerprinting")
        
        # Step 1: Preprocess data
        logger.info("Preprocessing data")
        processed_df = self.preprocess_data(df, price_col)
        
        # Step 2: Detect multi-scale extrema
        logger.info("Detecting extrema")
        processed_df['extrema'] = self.detect_extrema_multi_scale(processed_df, f"{price_col}_smooth")
        extrema_points = processed_df[processed_df['extrema'] != 0]
        peaks = extrema_points[extrema_points['extrema'] == 1]
        troughs = extrema_points[extrema_points['extrema'] == -1]
        print(f"DEBUG EXTREMA - Total: {len(extrema_points)}, Peaks: {len(peaks)}, Troughs: {len(troughs)}")
        
        # Step 3: Detect Cup and Handle patterns
        logger.info("Detecting Cup and Handle patterns")
        patterns, rejection_stats = self.detect_cup_and_handle(processed_df, 'extrema', f"{price_col}_smooth")
        
        # Step 4: Create fingerprints for all patterns
        logger.info("Creating pattern fingerprints")
        pattern_fingerprints = []
        for i, pattern in enumerate(patterns):
            fingerprint = self.create_pattern_fingerprint(pattern, processed_df, f"{price_col}_smooth")
            fingerprint["pattern_index"] = i + 1
            pattern_fingerprints.append(fingerprint)
            
            # Print key measurements for each pattern
            print(f"\n=== PATTERN {i+1} FINGERPRINT ===")
            print(f"ID: {fingerprint['pattern_id']}")
            print(f"Rim Height Diff: {fingerprint['rim_height_diff_pct']:.3f}%")
            print(f"Cup Depth: {fingerprint['cup_depth_pct']:.2f}% | Cup Width: {fingerprint['cup_width_minutes']:.0f}min")
            print(f"Cup Symmetry: {fingerprint['cup_symmetry']:.2f} | Cup Roundness: {fingerprint['cup_roundness']:.2f}")
            print(f"Handle Depth: {fingerprint['handle_depth_pct']:.2f}% | Handle Width: {fingerprint['handle_width_minutes']:.0f}min")
            print(f"Cup Aspect Ratio: {fingerprint['cup_aspect_ratio']:.1f} | Handle/Cup Duration: {fingerprint['handle_to_cup_duration_ratio']:.2f}")
            print(f"Quality Score: {fingerprint['quality_score']:.1f} | Shape: {'V-Cup' if fingerprint['is_cup_v_shaped'] else 'U-Cup' if fingerprint['is_cup_u_shaped'] else 'Medium-Cup'}")
            print(f"Handle Type: {'Steep' if fingerprint['is_handle_steep'] else 'Shallow' if fingerprint['is_handle_shallow'] else 'Medium'}")
        print(f"🔧 DEBUG: About to filter with min_cup_depth = {detector.config['min_cup_depth']}")
        for i, p in enumerate(patterns):
            cup_depth = p.get('cup_depth_pct', 0)
            should_pass = cup_depth >= (detector.config['min_cup_depth'] * 100)  # Convert to percentage
            print(f"  Pattern {i+1}: {cup_depth:.1f}% depth - Should pass: {should_pass}")
        # Step 5: Filter high-quality patterns
        logger.info("Filtering high-quality patterns")
        filtered_patterns = self.filter_patterns(patterns)
        
        # Create fingerprints for filtered patterns only
        filtered_fingerprints = []
        for i, pattern in enumerate(filtered_patterns):
            fingerprint = self.create_pattern_fingerprint(pattern, processed_df, f"{price_col}_smooth")
            fingerprint["pattern_index"] = i + 1
            fingerprint["is_high_quality"] = True
            filtered_fingerprints.append(fingerprint)
        
        # Return comprehensive results
        return {
            "all_patterns": patterns,
            "high_quality_patterns": filtered_patterns,
            "all_fingerprints": pattern_fingerprints,
            "high_quality_fingerprints": filtered_fingerprints,
            "rejection_stats": rejection_stats,
            "processed_df": processed_df
        }
    
    def __init___with_deduplication_fix(self, config=None):
        """Enhanced __init__ with deduplication tracking"""
        # Call original __init__ logic here (your existing config setup)
        
        # Add pattern tracking for deduplication
        self._created_patterns = set()
        self._pattern_fingerprints = {}
        
        print("🔧 STEP 2: Deduplication tracking initialized")
    
    def detect_cup_and_handle_deduped(self, df, extrema_col='extrema', price_col='close_smooth'):
        """
        STEP 2: Fixed version with proper deduplication
        Prevents creating multiple patterns from the same cup formation
        """
        
        # Reset stats for this detection run
        self.detection_stats = {
            'resistance_levels': {'total_found': 0, 'top_levels': [], 'price_range': None},
            'accumulation_analysis': {'levels_tested': 0, 'valid_accumulations': 0, 'rejection_reasons': {'no_bars_in_zone': 0, 'periods_too_short': 0, 'high_volatility': 0, 'large_moves': 0, 'volatility_increase': 0}},
            'handle_analysis': {'accumulations_tested': 0, 'handles_found': 0, 'rejection_reasons': {'insufficient_data': 0, 'depth_invalid': 0, 'duration_invalid': 0}},
            'pattern_creation': {'handles_tested': 0, 'breakouts_found': 0, 'patterns_created': 0}
        }
        
        patterns = []
        rejection_reasons = {'no_resistance_levels': 0, 'no_accumulation': 0, 'no_handle': 0}
        
        # STEP 2.1: Track created patterns to prevent duplicates
        created_cup_signatures = set()
        
        print(f"🏛️ Running DEDUPED institutional detection...")
        
        # Step 1: Find resistance levels
        resistance_levels = self.detect_resistance_levels(df)
        
        if not resistance_levels:
            rejection_reasons['no_resistance_levels'] = 1
            self.print_detection_summary()
            patterns = self.remove_duplicate_patterns(patterns)
            return patterns, rejection_reasons
        
        print(f"   📊 Found {len(resistance_levels)} resistance levels")
        
        # Step 2-4: Process each resistance level
        resistance_limit = self.config.get('process_resistance_limit', 200)
        print(f"   📊 Processing top {min(resistance_limit, len(resistance_levels))} resistance levels")
        for i, resistance in enumerate(resistance_levels[:resistance_limit]): # Limit to top 5 resistance levels
            print(f"\n   🔍 Processing resistance level {i+1}: ${resistance['price']:.2f}")
            
            accumulation_periods = self.detect_accumulation_base(df, resistance['price'])
            print(f"      Found {len(accumulation_periods)} accumulation periods")
            
            for j, accumulation in enumerate(accumulation_periods):
                print(f"      📋 Accumulation {j+1}: {accumulation['start']} to {accumulation['end']}")
                print(f"🔍 TESTING ACCUMULATION {j+1}: {accumulation['start']} to {accumulation['end']}")
                # STEP 2.2: Create unique signature for this cup formation
                # Use accumulation period + resistance level to identify unique cups
                week_bucket = accumulation['start'].strftime('%Y-%W')  # Group by year-week
                month_bucket = accumulation['start'].strftime('%Y-%m')  # Group by month instead of week
                cup_signature = f"{accumulation['start'].strftime('%Y%m%d_%H%M')}_{accumulation['end'].strftime('%H%M')}"
                  # Include day for uniqueness

       
            
                
                if not self.config.get('disable_deduplication', False) and cup_signature in created_cup_signatures:
                    # Check if this is a deep cup - if so, allow it through
                    acc_data = df.loc[accumulation['start']:accumulation['end']]
                    if len(acc_data) > 0:
                        rough_depth = (resistance['price'] - acc_data['low'].min()) / resistance['price'] * 100
                        if rough_depth >= 8.0:
                            print(f"         🎯 ALLOWING DEEP CUP: {rough_depth:.1f}% depth - bypassing deduplication")
                            # Create unique signature for deep cups to avoid conflicts
                            cup_signature = f"{cup_signature}_deep_{rough_depth:.1f}"
                            
                        else:
                            print(f"         ⏭️  SKIPPING: Cup already processed")
                            continue
                    else:
                        print(f"         ⏭️  SKIPPING: Cup already processed")
                        continue

                if cup_signature in created_cup_signatures:
                    print(f"         ⏭️  SKIPPING: Cup already processed")
                    continue

                created_cup_signatures.add(cup_signature) 

                cup_period = df.loc[accumulation['start']:accumulation['end']]
                trough_b_idx = cup_period['low'].idxmin()

                

                # We need a left rim price - let's find the best peak before accumulation
                acc_start_idx = df.index.get_loc(accumulation['start'])
                search_start = max(0, acc_start_idx - 96)
                left_peaks = df.iloc[search_start:acc_start_idx]
                left_peaks = left_peaks[left_peaks['extrema'] == 1]

                if len(left_peaks) > 0:
                    # Find HIGHEST peak that's reasonably close to resistance level
                    best_left_rim = None
                    best_left_rim_price = 0  # Start with 0 to find highest
                    
                    for peak_time, peak_row in left_peaks.iterrows():
                        peak_price = peak_row['high']
                        # Must be reasonably close to resistance (within 10% for ES futures)
                        if abs(peak_price - resistance['price']) / resistance['price'] <= 0.10:
                            if peak_price > best_left_rim_price:  # Find HIGHEST, not closest
                                best_left_rim_price = peak_price
                                best_left_rim = peak_price
                                best_left_rim_time = peak_time
                    
                    if best_left_rim is not None:
                        peak_a_price = best_left_rim
                        peak_a_time = best_left_rim_time
                        peak_c_time = self.find_optimal_right_rim(
                            df,
                            accumulation['start'],
                            accumulation['end'],
                            resistance['price'],    # resistance_level
                            best_left_rim,          # left_rim_price
                            trough_b_idx            # trough_time
                        )
                    else:
                        continue  # Skip this accumulation if no left rim found
                else:
                    continue  # Skip this accumulation if no left rim found


                if peak_c_time is None or pd.isna(peak_c_time):
                    print(f"   ❌ Invalid peak_c_time: {peak_c_time}")
                    continue

                if peak_c_time not in df.index:
                    print(f"   ❌ peak_c_time {peak_c_time} not in DataFrame index")
                    continue

                peak_c_price = df.loc[peak_c_time, 'high']

                if abs(peak_c_price - resistance['price']) / resistance['price'] > 0.05:
                     continue 
                
                rim_valid, rim_msg = self.validate_no_higher_highs_in_cup(
                    df, peak_a_time, peak_c_time, tolerance_pct=2.0
                )
                if not rim_valid:
                    print(f"         ❌ {rim_msg}")
                    continue

                trough_b_time = trough_b_idx     
                handles = self.detect_handle_formation(df, resistance['price'], peak_c_time, peak_c_price, peak_a_time, trough_b_time)
                print(f"         Found {len(handles)} potential handles")
                
                # STEP 2.3: Only use the BEST handle for each cup (not all variations)
                if handles:
                    # Sort handles by quality score and pick the best one
                    best_handle = max(handles, key=lambda h: h['score'])
                    print(f"         🎯 Selected best handle: score={best_handle['score']}")

                    breakout_time = self.find_breakout_after_handle(
                        df, 
                        best_handle['end'], 
                        resistance['price'], 
                        peak_a_price, 
                        peak_c_price, 
                        peak_c_time
                    )

                    pattern = self.create_institutional_pattern(
                        resistance, accumulation, best_handle, breakout_time, df, peak_c_time, price_col
                    )
                    
                    if pattern:
                        patterns.append(pattern)
                        print(f"         ✅ PATTERN CREATED!")
                    else:
                        print(f"         ❌ Pattern creation failed")

                 
        
        # Print summary at the end
        print(f"\n📊 STEP 2 DEDUPLICATION RESULTS:")
        print(f"   Unique cup signatures processed: {len(created_cup_signatures)}")
        print(f"   Final patterns created: {len(patterns)}")
        
        self.print_detection_summary()
        return patterns, rejection_reasons
    
    def detect_formation_first(self, df, extrema_col='extrema', price_col='close_smooth'):
        """
        Formation-first detection: Add this method after detect_cup_and_handle_deduped
        """
        print(f"\n🔍 FORMATION-FIRST DETECTION (complementary to institutional):")
        
        patterns = []
        peaks = df[df[extrema_col] == 1]
        troughs = df[df[extrema_col] == -1]
        
        if len(peaks) < 3 or len(troughs) < 2:
            print(f"   ❌ Insufficient extrema for formation-first")
            return []
        
        print(f"   📊 Processing {len(peaks)} peaks, {len(troughs)} troughs")
        
        # Process in small batches to avoid memory issues
        batch_size = 30  # Small batches
        
        for i, peak_a_time in enumerate(peaks.index[:-2]):  # Need at least 2 more peaks
            if i >= batch_size:  # Limit total processing
                print(f"   ⏹️ Stopping at {batch_size} peaks to prevent overload")
                break
                
            peak_a_price = df.loc[peak_a_time, 'high']
            
            # Find troughs after this peak (max 5 to keep it manageable)
            candidate_troughs = troughs[troughs.index > peak_a_time][:5]
            
            for trough_b_time in candidate_troughs.index:
                trough_b_price = df.loc[trough_b_time, 'low']
                
                # Find peaks after this trough (max 3 to keep it manageable)
                candidate_peaks = peaks[peaks.index > trough_b_time][:3]
                
                for peak_c_time in candidate_peaks.index:
                    peak_c_price = df.loc[peak_c_time, 'high']
                    
                    # Quick validation before expensive operations
                    if not self._quick_cup_check(peak_a_price, trough_b_price, peak_c_price, 
                                                peak_a_time, peak_c_time):
                        continue
                    
                    # Simple handle detection
                    handle = self._find_formation_handle(df, peak_c_time, peak_c_price)
                    if not handle:
                        continue
                    
                    # Simple breakout detection
                    breakout_time = self._find_formation_breakout(df, handle, max(peak_a_price, peak_c_price))
                    if not breakout_time:
                        continue
                    
                    # Create pattern
                    pattern = self._create_formation_pattern(df, peak_a_time, trough_b_time, 
                                                           peak_c_time, handle, breakout_time, price_col)
                    if pattern:
                        patterns.append(pattern)
                        print(f"      ✅ Formation pattern: {peak_a_time.strftime('%m-%d %H:%M')}")
        
        print(f"   📊 Formation-first found: {len(patterns)} additional patterns")
        return patterns

    
    def adjust_pattern_points(self, pattern, df, price_col='close_smooth'):
        """Fine-tune pattern points to ensure ideal cup and handle geometry."""
        print(f"🔧 ADJUSTING PATTERN: A={pattern['peak_a']}, C={pattern['peak_c']}")
        try:
            # Get timestamps
            peak_a_ts = pattern["peak_a"]
            peak_c_ts = pattern["peak_c"]
            trough_b_ts = pattern["trough_b"]
            
            # Get prices of both rims
            price_a = df.loc[peak_a_ts, price_col]
            price_c = df.loc[peak_c_ts, price_col]

        
                        
            # Check if rim heights already meet our $0.25 absolute tolerance
            rim_diff_abs = abs(price_a - price_c)
            rim_diff_pct = rim_diff_abs / max(price_a, price_c) * 100
            tolerance_pct = self.config.get('rim_height_tolerance_pct', 0.3)

            if rim_diff_pct <= tolerance_pct:  # Use absolute tolerance, not percentage
                print(f"✅ NO ADJUSTMENT NEEDED: rim diff {rim_diff_pct:.2f}% <= {tolerance_pct}%")
                logger.info(f"Pattern accepted without adjustment: rim diff ${rim_diff_abs:.2f}")
                return pattern
                
            # Get optimal rim height (use the higher of the two rims)
            optimal_rim_height = max(price_a, price_c)
            
            # Determine which rim needs adjustment (always adjust the lower one)
            if price_a < price_c:
                # Look for a better point for rim A
                # Expand search window to 10 bars (or more) before and after
                window = 10
                a_idx = df.index.get_loc(peak_a_ts)
                b_idx = df.index.get_loc(trough_b_ts)
                
                # Only search between cup start and bottom
                start_idx = max(0, a_idx - window)
                end_idx = min(b_idx, a_idx + window)
                
                best_diff = abs(price_a - optimal_rim_height)
                best_idx = a_idx
                
                # First try to find extrema points
                for i in range(start_idx, end_idx):
                    if i < len(df) and df.iloc[i]['extrema'] == 1:
                        curr_price = df.iloc[i][price_col]
                        curr_diff = abs(curr_price - optimal_rim_height)
                        if curr_diff < best_diff:
                            best_diff = curr_diff
                            best_idx = i
                
                # If we couldn't find a suitable extrema, try high points
                if best_idx == a_idx and best_diff > 0.001:
                    for i in range(start_idx, end_idx):
                        # Check if it's a local high point
                        if i > 0 and i < len(df) - 1:
                            curr_price = df.iloc[i][price_col]
                            # Check if higher than neighbors and closer to optimal height
                            if (curr_price >= df.iloc[i-1][price_col] and 
                                curr_price >= df.iloc[i+1][price_col]):
                                curr_diff = abs(curr_price - optimal_rim_height)
                                if curr_diff < best_diff:
                                    best_diff = curr_diff
                                    best_idx = i
                
                if best_idx != a_idx:
                    # Found a better point
                    new_a_time = df.index[best_idx]
                    original_b = pattern.get("trough_b")
                    
                    # Check if adjustment would break chronological order
                    if original_b and new_a_time >= original_b:
                        print(f"❌ REVERTING left rim adjustment - would break chronological order (new A {new_a_time} >= B {original_b})")
                        return None  # Signal to skip this pattern
                    
                    # Safe to adjust A
                    pattern["peak_a"] = new_a_time
                    logger.info(f"Adjusted left rim from {peak_a_ts} to {pattern['peak_a']} to match rim heights")
                    
                    
            elif price_c < price_a:
                # Look for a better point for rim C
                window = 10
                c_idx = df.index.get_loc(peak_c_ts)
                b_idx = df.index.get_loc(trough_b_ts)
                
                # Only search between cup bottom and right rim
                start_idx = max(b_idx, c_idx - window)
                end_idx = min(len(df), c_idx + window)
                
                best_diff = abs(price_c - optimal_rim_height)
                best_idx = c_idx
                
                # First try to find extrema points
                for i in range(start_idx, end_idx):
                    if i < len(df) and df.iloc[i]['extrema'] == 1:
                        curr_price = df.iloc[i][price_col]
                        curr_diff = abs(curr_price - optimal_rim_height)
                        if curr_diff < best_diff:
                            best_diff = curr_diff
                            best_idx = i
                
                # If we couldn't find a suitable extrema, try high points
                if best_idx == c_idx and best_diff > 0.001:
                    for i in range(start_idx, end_idx):
                        # Check if it's a local high point
                        if i > 0 and i < len(df) - 1:
                            curr_price = df.iloc[i][price_col]
                            # Check if higher than neighbors and closer to optimal height
                            if (curr_price >= df.iloc[i-1][price_col] and 
                                curr_price >= df.iloc[i+1][price_col]):
                                curr_diff = abs(curr_price - optimal_rim_height)
                                if curr_diff < best_diff:
                                    best_diff = curr_diff
                                    best_idx = i
                
                if best_idx != c_idx:
                     # Found a better point
                    new_c_time = df.index[best_idx]
                    original_d = pattern.get('handle_d')
                    
                    print(f"🚨 RIM ADJUSTMENT: C was {peak_c_ts}, D was {original_d}, D>C before: {original_d > peak_c_ts if original_d else 'N/A'}")
                    
                    # Check if adjustment would break chronological order
                    if original_d and new_c_time >= original_d:
                        print(f"❌ REVERTING rim adjustment - would break chronological order (new C {new_c_time} >= D {original_d})")
                        return None  # Signal to skip this pattern
                    
                    # Safe to adjust C
                    pattern["peak_c"] = new_c_time
                    logger.info(f"Adjusted right rim from {peak_c_ts} to {pattern['peak_c']} to match rim heights")
            
            
            # Force exact rim height adjustment if we couldn't find points naturally
            # Recalculate final prices
            final_price_a = df.loc[pattern["peak_a"], price_col]
            final_price_c = df.loc[pattern["peak_c"], price_col]

            try:
                if "handle_d" not in pattern:
                    logger.warning("Cannot recalculate breakout: 'handle_d' not found in pattern")
                    pattern["breakout_time"] = None
                else:
                    # Ensure breakout starts after both peak_c and trough_d
                    cup_end_idx = max(
                        df.index.get_loc(pattern["peak_c"]),
                        df.index.get_loc(pattern["handle_d"])
                    )
                    final_rim_price = max(
                        df.loc[pattern["peak_a"], price_col],
                        df.loc[pattern["peak_c"], price_col]
                    )
                    breakout_idx = None

                    for i in range(cup_end_idx + 1, min(len(df), cup_end_idx + 40)):
                        if df.iloc[i][price_col] > final_rim_price:
                            breakout_idx = i
                            break

                    if breakout_idx is not None:
                        pattern["breakout_time"] = df.index[breakout_idx]
                    else:
                        logger.info(f"No breakout found after rim adjustment for pattern ending at {df.index[cup_end_idx]}")
                        pattern["breakout_time"] = None
            except Exception as e:
                logger.warning(f"Error recalculating breakout after rim adjustment: {e}")

            
            # If heights still differ by more than 0.2%, store an adjustment factor for visualization
            # Calculate missing variables for validation
            final_rim_diff = abs(final_price_a - final_price_c)
            # tolerance_abs = self.config.get('rim_height_tolerance_abs', 0.25)
            final_rim_diff_pct = final_rim_diff / max(final_price_a, final_price_c) * 100

            if final_rim_diff_pct > tolerance_pct:
                print(f"❌ REJECTING adjusted pattern: rim diff ${final_rim_diff:.2f} > ${tolerance_pct}")
                logger.info(f"Pattern rejected for rim height violation: ${final_rim_diff:.2f} > ${tolerance_pct}")
                return None  # Reject the pattern entirely

            print(f"✅ KEEPING adjusted pattern: rim diff ${final_rim_diff:.2f} <= ${tolerance_pct}")
            logger.info(f"Pattern ACCEPTED: rim diff ${final_rim_diff:.2f} <= ${tolerance_pct}")

            # If heights still differ by more than 0.2%, store an adjustment factor for visualization  
            if abs(final_price_a - final_price_c) / max(final_price_a, final_price_c) > 0.002:
                if final_price_a < final_price_c:
                    pattern["adjusted_a_price"] = final_price_c
                else:
                    pattern["adjusted_c_price"] = final_price_a
        
        except Exception as e:
            logger.warning(f"Error adjusting pattern points: {e}")
            
            # Still validate rim height even if adjustment failed
            final_price_a = df.loc[pattern["peak_a"], price_col]
            final_price_c = df.loc[pattern["peak_c"], price_col]
            final_rim_diff = abs(final_price_a - final_price_c)
            # tolerance_abs = self.config.get('rim_height_tolerance_abs', 0.25)

            # if final_rim_diff > tolerance_abs:
            #     print(f"❌ REJECTING pattern (exception path): rim diff ${final_rim_diff:.2f} > ${tolerance_abs}")
            #     return None  # Reject the entire pattern

            return pattern

    def analyze_breakout_failures(self):
                    """Analyze why breakouts are failing - call this after detection"""
                    print(f"\n🔍 BREAKOUT FAILURE ANALYSIS:")
                    
                    # Get some sample resistance levels and check what happens after handles
                    r_stats = self.detection_stats['resistance_levels']
                    if r_stats['top_levels']:
                        sample_resistance = r_stats['top_levels'][0]['price']  # Top resistance level
                        print(f"   Analyzing top resistance level: ${sample_resistance:.2f}")
                        
                        # This would require storing handle data, but gives the idea
                        print(f"   Common reasons for breakout failure in ES futures:")
                        print(f"   1. Resistance levels too precise (penny-perfect)")
                        print(f"   2. Market gaps over resistance without touching it") 
                        print(f"   3. Handle formation at end of data (no future bars)")
                        print(f"   4. Resistance level calculated from old data")
                        print(f"   5. Need to allow 'near-miss' breakouts (within 0.1-0.5%)")

    def validate_chronological_order(self, peak_a, trough_b, peak_c, handle_d, breakout_e):
        """Ensure proper chronological order: A < B < C < D < E"""
        timestamps = [peak_a, trough_b, peak_c, handle_d, breakout_e]
        
        for i in range(len(timestamps) - 1):
            if timestamps[i] >= timestamps[i + 1]:
                return False, f"Chronological violation: {timestamps[i]} >= {timestamps[i+1]}"
        
        return True, "Valid chronological order"

    
    def validate_handle_position_correct(self, df, peak_c_ts, handle_d_ts, trough_b_ts):
        """FIXED handle validation using human-labeled criteria"""
        
        peak_c_price = df.loc[peak_c_ts, 'high']
        handle_d_price = df.loc[handle_d_ts, 'low'] 
        trough_b_price = df.loc[trough_b_ts, 'low']
        
        # RULE 1: Handle must NEVER go below cup bottom
        tolerance_pct = 0.005  # 0.5% tolerance for noise
        if handle_d_price < (trough_b_price * (1 - tolerance_pct)):
            below_distance = trough_b_price - handle_d_price
            print(f"REJECTED: Handle ${handle_d_price:.2f} below cup bottom ${trough_b_price:.2f} by ${below_distance:.2f}")
            return False, "handle_below_cup_bottom"
        
        # Calculate actual handle drop
        handle_drop_points = abs(peak_c_price - handle_d_price)
        handle_drop_pct = (handle_drop_points / peak_c_price) * 100

        # RULE 2: Handle depth based on HUMAN LABELS (0.30-0.65% valid range)
        timeframe_minutes = self.detect_timeframe(df) if hasattr(self, 'detect_timeframe') else 15
        
        if timeframe_minutes <= 15:
            # Based on human-labeled valid patterns
            min_handle_pct = 0.30  # ← FROM HUMAN LABELS: minimum valid
            max_handle_pct = 0.65  # ← FROM HUMAN LABELS: maximum valid
            print(f"🎯 Using human-derived criteria: {min_handle_pct}-{max_handle_pct}%")
        else:
            # Keep existing for other timeframes
            min_handle_pct = self.config.get('min_handle_depth_pct', 0.3)
            max_handle_pct = 25.0

        # Validation
        if handle_drop_pct < min_handle_pct:
            print(f"❌ Handle too shallow: {handle_drop_pct:.2f}% < {min_handle_pct}% (human criteria)")
            return False, "handle_too_shallow"
        
        if handle_drop_pct > max_handle_pct:
            print(f"❌ Handle too deep: {handle_drop_pct:.2f}% > {max_handle_pct}% (human criteria)")
            return False, "handle_too_deep"
        
        print(f"✅ VALID HANDLE: {handle_drop_pct:.2f}% within human range {min_handle_pct}-{max_handle_pct}%")
        return True, "valid_handle_position"
    

    def is_significant_peak(self, df, peak_time, atr_multiplier=1.5):
        """Simple ATR significance check"""
        try:
            atr_series = self.calculate_atr(df, 20)
            peak_idx = df.index.get_loc(peak_time)
            current_atr = atr_series.iloc[peak_idx]
            peak_price = df.loc[peak_time, 'high']
            
            # Simple prominence check
            window = df.iloc[max(0, peak_idx-5):peak_idx+6]
            prominence = peak_price - window['low'].min()
            required = current_atr * atr_multiplier
            
            return prominence >= required
        except:
            return True  
    
    def _validate_formation_geometry(self, df, peak_a_time, trough_b_time, peak_c_time):
        """Basic geometric validation for cup formations."""
        
        peak_a_price = df.loc[peak_a_time, 'high'] 
        trough_b_price = df.loc[trough_b_time, 'low']
        peak_c_price = df.loc[peak_c_time, 'high']
        
        # 1. Cup depth check (1-30%)
        cup_depth_pct = (peak_a_price - trough_b_price) / peak_a_price * 100
        if not (1.0 <= cup_depth_pct <= 30.0):
            return False
        
        # 2. Rim symmetry check (within 5%)
        rim_diff_pct = abs(peak_a_price - peak_c_price) / max(peak_a_price, peak_c_price) * 100
        if rim_diff_pct > 5.0:
            return False
        
        # 3. Duration check (3 hours to 30 days)
        cup_duration_hours = (peak_c_time - peak_a_time).total_seconds() / 3600
        if not (3 <= cup_duration_hours <= 48):
            return False
        
        # 4. Ensure trough is actually lower than both peaks
        if trough_b_price >= min(peak_a_price, peak_c_price):
            return False
        
        return True
    
    def get_timeframe_specific_config(self, timeframe_minutes):
        """Get configuration adapted to specific timeframe"""
        
        if timeframe_minutes >= 1440:  # Daily+ data
            return {
                "rim_height_tolerance_pct": 5.0,
                "min_handle_drop": 1.5,
                "min_handle_drop_config": 1.5,
                "min_cup_depth": 0.015,
                "min_cup_duration": self.config.get('min_cup_duration', 2880) * 2.5,  # Scale from base
                "max_cup_duration": self.config.get('max_cup_duration', 14400) * 9,
            }
        elif timeframe_minutes >= 240:  # 4H data
            return {
                "rim_height_tolerance_pct": 8.0,
                "min_handle_depth_pct": 0.8, 
                "min_handle_drop_config": 0.8,
                "min_cup_depth": 0.015,
                "min_cup_duration": self.config.get('min_cup_duration', 2880),  # Use base config
                "max_cup_duration": self.config.get('max_cup_duration', 14400),
            }
        elif timeframe_minutes >= 60:   # 1H data
            return {
                "rim_height_tolerance_pct": 10.0,
                "min_handle_depth_pct": 0.5,
                "min_handle_drop_config": 0.5,
                "min_cup_depth": 0.04,
                "min_cup_duration": self.config.get('min_cup_duration', 2880),
                "max_cup_duration": self.config.get('max_cup_duration', 14400),
            }
        else:  # Intraday (15min, 30min)
             return {
                    "rim_height_tolerance_pct": 12.0,
                    "min_handle_depth_pct": 0.30,          # ← FROM 0.3 TO 0.30 (human labels)
                    "max_handle_depth_pct": 0.65,          # ← ADD THIS NEW LINE (human labels)
                    "min_handle_drop_config": 0.01,
                    "min_cup_depth": 0.006,                # ← FROM 0.015 TO 0.006 (0.6% from human labels)
                    "min_cup_duration": 300,               # ← 5 hours minimum (human labels)
                    "max_cup_duration": 1320,        
                }


def create_pattern_review_report(fingerprints, output_dir="cup_handle_results"):
    """Create an interactive report for pattern review and labeling."""
    
    # Create a CSV for easy pattern review
    df_fingerprints = pd.DataFrame(fingerprints)
    # Handle case where no patterns were detected
    if len(df_fingerprints) == 0:
        print(f"\n=== NO PATTERNS DETECTED ===")
        print(f"No patterns found to create review report.")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Add columns for manual labeling
    df_fingerprints['is_true_cup_handle'] = ""  # You'll fill this: YES/NO
    df_fingerprints['notes'] = ""  # Your comments about why it's good/bad
    df_fingerprints['confidence_level'] = ""  # HIGH/MEDIUM/LOW
    
    # Select key columns for review
    review_columns = [
        'pattern_id', 'pattern_index', 'is_true_cup_handle', 'confidence_level', 'notes',
        'rim_height_diff_pct', 'cup_depth_pct', 'cup_width_minutes', 
        'cup_symmetry', 'cup_roundness', 'cup_aspect_ratio',
        'handle_depth_pct', 'handle_width_minutes', 'handle_to_cup_duration_ratio',
        'quality_score', 'is_cup_v_shaped', 'is_cup_u_shaped', 
        'is_handle_steep', 'is_handle_shallow'
    ]
    
    review_df = df_fingerprints[review_columns].copy()
    review_df.to_csv(os.path.join(output_dir, "pattern_review.csv"), index=False)
    
    print(f"\n=== PATTERN REVIEW SYSTEM ===")
    print(f"1. Open: {output_dir}/pattern_review.csv")
    print(f"2. Fill in the 'is_true_cup_handle' column with YES/NO for each pattern")
    print(f"3. Add notes about what makes each pattern good/bad")
    print(f"4. Save the file and run the learning function")
    print(f"\nPattern Summary:")
    for i, fp in enumerate(fingerprints):
        print(f"Pattern {i+1} ({fp['pattern_id']}): Cup {fp['cup_depth_pct']:.1f}%, Handle {fp['handle_depth_pct']:.1f}%, Quality {fp['quality_score']:.1f}")
    
    return review_df
        
            # Main execution function (REPLACE the existing run_cup_handle_detection function)
def run_detection_with_review(data_path, output_dir="cup_handle_results", config=None):
    """Run detection and create review system."""
    
    # Create detector
    detector = CupHandleDetector(config)
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    df.columns = [col.lower() for col in df.columns]
    
    # Run detection with fingerprinting
    detection_results = detector.detect_with_fingerprints(df)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save fingerprints
    if detection_results["all_fingerprints"]:
        fingerprints_df = pd.DataFrame(detection_results["all_fingerprints"])
        fingerprints_df.to_csv(os.path.join(output_dir, "pattern_fingerprints.csv"), index=False)
    
    # Create pattern review system
    create_pattern_review_report(detection_results["all_fingerprints"], output_dir)
    
    # Generate visualizations
    if detection_results["high_quality_patterns"]:
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        for i, pattern in enumerate(detection_results["all_patterns"]):
            viz_path = os.path.join(viz_dir, f"pattern_{i+1}.png")
            detector.visualize_pattern(pattern, detection_results["processed_df"], viz_path)

    return detection_results


        # Add this helper function to analyze pattern similarities
def compare_patterns(fingerprint1, fingerprint2, weights=None):
    """Compare two pattern fingerprints and return similarity score."""
    if weights is None:
        weights = {
            'rim_height_diff_pct': 0.15,
            'cup_depth_pct': 0.20,
            'cup_symmetry': 0.15,
            'cup_roundness': 0.10,
            'handle_depth_pct': 0.15,
            'cup_aspect_ratio': 0.10,
            'handle_to_cup_duration_ratio': 0.15
        }
    
    similarity = 0
    for key, weight in weights.items():
        if key in fingerprint1 and key in fingerprint2:
            # Normalize difference to 0-1 scale
            val1, val2 = fingerprint1[key], fingerprint2[key]
            max_val = max(abs(val1), abs(val2), 1)  # avoid division by zero
            diff = abs(val1 - val2) / max_val
            similarity += weight * (1 - min(diff, 1))  # convert difference to similarity
    
    return similarity


if __name__ == "__main__":
    

    input_path = "ES_timeframes/ES_15min.csv"
    output_path = "detected_patterns.json"

    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    print(f"📂 Loading data from {input_path}")



    custom_config = {
    # Ultra-relaxed detection
    "rim_height_tolerance_pct": 1.0,
    "min_cup_roundness": 0.3,
    "min_cup_symmetry": 0.15,
    "min_quality_score": 55,
    "min_cup_depth": 0.003,
    "breakout_max_above_rim_pct": 0.5,
    "disable_deduplication": True,
    "enable_formation_first": True,
    "min_handle_gap_minutes": 150, 
    "max_handle_depth_pct": 25.0,  # Keep existing max
    "min_handle_depth_pct": 0.05,   # ✅ FIXED: Reduced from 2.0 to 0.3 for ES futures
    "breakout_minimum_pct": 100.2,
    "handle_must_be_below_rim_pct": 2.0,  
    "breakout_tolerance_pct": 0.2,
    "max_price_above_rim_during_cup_pct": 0.1, 
    "min_handle_zone_duration_minutes": 150,
    "breakout_minimum_above_rim_pct": 0.2,
    "breakout_minimum_above_rim_points": 5.0,    
    # Processing limits - INCREASED
    "process_resistance_limit": 500,    
    "max_resistance_bars": 250000,   
    "min_cup_atr_multiple": 4.0,        
    "min_handle_atr_multiple": 1.5,     
    "atr_period": 20,                   
    # Durations
    "min_cup_duration": 60,
    "max_cup_duration": 2880,
    "min_handle_duration": 30,
    "max_handle_duration": 360,
    "breakout_search_duration": 480,
    "rim_search_duration": 1440,
    "handle_search_duration": 960,
    "max_handle_drop": 25.0,
    "use_hybrid_detection": True,      
    "enable_atr_filtering": True,    
    "atr_multiplier": 1.5,           
    # Keep working settings
    "skip_rim_adjustment": True,
}

     
    df = pd.read_csv(input_path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df.set_index('timestamp', inplace=True)

    # 🔥 FILTER: Only last 10 years to avoid massive time gaps
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=6)
    original_length = len(df)
    df = df[df.index >= cutoff_date]
    print(f"📅 FILTERED DATA: {original_length} → {len(df)} bars (last 10 years since {cutoff_date.date()})")

    detector = CupHandleDetector(config=custom_config)
    print(f"🔍 ACTUAL CONFIG CHECK: min_cup_duration = {detector.config['min_cup_duration']}")
    print(f"🔍 ACTUAL CONFIG CHECK: max_gap_bars would be = {90 if detector.detect_timeframe(df) >= 1440 else 30}")
    detector.detect_cup_and_handle_original = detector.detect_cup_and_handle
    detector.detect_cup_and_handle = detector.detect_cup_and_handle_deduped
    combined_results = detector.detect_combined(df)
    patterns = combined_results['all_patterns']
    # Print analysis of both systems
    print(f"\n🔍 COMBINED DETECTION RESULTS:")
    print(f"   Strict patterns: {len(combined_results['strict_patterns'])}")
    print(f"   Relaxed patterns: {len(combined_results['relaxed_patterns'])}")
    print(f"   Total patterns: {len(combined_results['all_patterns'])}")

    overlap_stats = combined_results['overlap_analysis']
    print(f"\n📊 OVERLAP ANALYSIS:")
    print(f"   Strict only: {overlap_stats['strict_only']}")
    print(f"   Relaxed only: {overlap_stats['relaxed_only']}")
    print(f"   Overlapping: {overlap_stats['overlapping']}")
    print(f"   Unique periods: {overlap_stats['unique_periods']}")

    # Show breakdown by source
    print(f"\n🏷️  PATTERN SOURCE BREAKDOWN:")
    for pattern in patterns[:10]:  # Show first 10
        source = pattern.get('detection_source', 'unknown')
        quality = pattern.get('quality_score', 0)
        pattern_id = pattern.get('pattern_id', 'no_id')
        print(f"   {pattern_id}: {source} (quality: {quality:.1f})")

    if len(patterns) == 0:
        print("⚠️ No patterns found from either system")

    # Count pattern types for verification
    v_shapes = [p for p in patterns if p.get('cup_roundness', 0) < 0.7]
    u_shapes = [p for p in patterns if p.get('cup_roundness', 0) >= 0.75]
    medium_shapes = [p for p in patterns if 0.7 <= p.get('cup_roundness', 0) < 0.75]

    print(f"\n📊 PATTERN TYPE BREAKDOWN:")
    print(f"   V-shaped cups (roundness < 0.7): {len(v_shapes)}")
    print(f"   Medium cups (0.7-0.75): {len(medium_shapes)}")  
    print(f"   U-shaped cups (≥0.75): {len(u_shapes)}")
    print(f"   Total: {len(patterns)} patterns")

    # Show supplementary patterns specifically
    supp_patterns = [p for p in patterns if p.get('detection_phase') == 'supplementary_u_shape']
    print(f"   New U-shapes found: {len(supp_patterns)}")

    
        
    with open(output_path, "w") as f:
        json.dump(patterns, f, indent=2, default=str)
    print(f"✅ Done! Saved to {output_path}")