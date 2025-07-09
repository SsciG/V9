def detect_all_patterns(self, df, price_col='close'):
        """
        Combined detection method that finds both V-shapes and U-shapes.
        Preserves your existing 21 patterns and adds high-quality U-shapes.
        """
        
        # Step 1: Run your existing successful detection
        print(f"🔍 PHASE 1: Running existing detection (finds V and some U shapes)...")
        existing_patterns = self.detect(df, price_col=price_col)
        
        v_shapes = [p for p in existing_patterns if p.get('cup_roundness', 0) < 0.7]
        u_shapes = [p for p in existing_patterns if p.get('cup_roundness', 0) >= 0.7]
        
        print(f"   Phase 1 Results: {len(existing_patterns)} total")
        print(f"   - V-shaped: {len(v_shapes)}")
        print(f"   - U-shaped: {len(u_shapes)}")
        
        # Step 2: Run U-shape focused detection for missed patterns
        print(f"\n🔍 PHASE 2: U-shape focused detection...")
        
        # Temporarily replace roundness calculation with enhanced version
        original_method = self.calculate_cup_roundness
        self.calculate_cup_roundness = self.calculate_cup_roundness
        
        try:
            new_u_patterns, u_rejections = self.detect_u_shaped_patterns(df, existing_patterns)
            
            print(f"   Phase 2 Results: {len(new_u_patterns)} new U-shapes found")
            
            # Step 3: Combine results
            all_patterns = existing_patterns + new_u_patterns
            
            # Sort by chronological order
            all_patterns.sort(key=lambda p: p['peak_a'])
            
            print(f"\n📊 FINAL COMBINED RESULTS:")
            print(f"   Total patterns: {len(all_patterns)}")
            print(f"   V-shapes: {len(v_shapes)}")
            print(f"   U-shapes (existing): {len(u_shapes)}")
            print(f"   U-shapes (new): {len(new_u_patterns)}")
            print(f"   Total U-shapes: {len(u_shapes) + len(new_u_patterns)}")
            
            return all_patterns
            
        finally:
            # Restore original roundness calculation
            self.calculate_cup_roundness = original_method



def detect_supplementary_u_shapes(self, df, existing_patterns, price_col='close_smooth'):
    """Find additional U-shaped patterns using relaxed constraints."""
    print(f"\n🔍 PHASE 2: Supplementary U-shape detection...")


        # FIX: Use the already processed dataframe that has volatility column
        # Instead of calling detect_cup_and_handle with raw df, we need processed df
        
        # Check if df has volatility column, if not, preprocess it
    if 'volatility' not in df.columns:
        print("   📊 Preprocessing data for U-shape detection...")
        processed_df = self.preprocess_data(df, price_col.replace('_smooth', ''))
        # Ensure extrema column exists
        if 'extrema' not in processed_df.columns:
            processed_df['extrema'] = self.detect_extrema_multi_scale(processed_df, price_col)
    else:
        processed_df = df  # Already processed
    
    # Run detection with new settings on processed data
    u_patterns, _ = self.detect_cup_and_handle(processed_df, 'extrema', price_col)
    
    # Filter for non-overlapping high-roundness patterns
    new_u_patterns = []
    existing_spans = [(p['peak_a'], p['breakout_e']) for p in existing_patterns]
    
    for pattern in u_patterns:
        # Check roundness threshold
        if pattern['cup_roundness'] < 0.75:
            continue
            
        # Check for overlap with existing patterns
        pattern_span = (pattern['peak_a'], pattern['breakout_e'])
        overlaps = any(
            (pattern_span[0] <= end and pattern_span[1] >= start)
            for start, end in existing_spans
        )
        
        if not overlaps:
            pattern['detection_phase'] = 'supplementary_u_shape'
            new_u_patterns.append(pattern)
            print(f"   ✅ Found U-shape: roundness={pattern['cup_roundness']:.2f}")
    
    print(f"   📊 Phase 2 Results: {len(new_u_patterns)} new U-shapes")
    return new_u_patterns


def detect_formations_direct(self, df, extrema_col='extrema'):
        """Find cup formations directly - ONE best formation per Peak A."""
        formations = []
        peaks = df[df[extrema_col] == 1]
        troughs = df[df[extrema_col] == -1]
        
        print(f"\n🔍 FORMATION-FIRST DETECTION:")
        print(f"   Available: {len(peaks)} peaks, {len(troughs)} troughs")
        
        formations_tested = 0
        formations_found = 0
        
        # Find peak A → BEST peak C combinations
        for i, peak_a_time in enumerate(peaks.index):
            peak_a_price = df.loc[peak_a_time, 'high']
            
            best_formation = None
            best_score = 0
            
            # Find troughs after this peak
            later_troughs = troughs[troughs.index > peak_a_time]
            
            for trough_b_time in later_troughs.index:
                trough_b_price = df.loc[trough_b_time, 'low']
                
                # Find peaks after this trough  
                later_peaks = peaks[peaks.index > trough_b_time]
                
                for peak_c_time in later_peaks.index:
                    peak_c_price = df.loc[peak_c_time, 'high']
                    
                    formations_tested += 1
                    
                    # Basic validation for cup geometry
                    if self._validate_formation_geometry(df, peak_a_time, trough_b_time, peak_c_time):
                        
                        # Score this formation (rim symmetry)
                        rim_diff = abs(peak_a_price - peak_c_price) / max(peak_a_price, peak_c_price)
                        symmetry_score = 1.0 - rim_diff  # Higher = more symmetric
                        
                        if symmetry_score > best_score:
                            best_score = symmetry_score
                            best_formation = {
                                'peak_a_time': peak_a_time,
                                'peak_a_price': peak_a_price,
                                'trough_b_time': trough_b_time, 
                                'trough_b_price': trough_b_price,
                                'peak_c_time': peak_c_time,
                                'peak_c_price': peak_c_price,
                                'natural_resistance': max(peak_a_price, peak_c_price),
                                'symmetry_score': symmetry_score
                            }
                    
                    # Limit search scope to prevent explosion
                    if len(later_peaks) > 20:
                        break
            
            # Add the BEST formation for this Peak A (if any)
            if best_formation:
                formations.append(best_formation)
                formations_found += 1
                
                print(f"   ✅ Formation {formations_found}: {peak_a_time.strftime('%m-%d %H:%M')} → {best_formation['peak_c_time'].strftime('%m-%d %H:%M')} (${best_formation['natural_resistance']:.2f}, sym: {best_formation['symmetry_score']:.2f})")
            
            # Progress indicator  
            if i % 50 == 0 and i > 0:
                print(f"   📊 Processed {i}/{len(peaks)} peaks...")
        
        print(f"   📊 Formation Detection Results:")
        print(f"      Tested: {formations_tested} A-B-C combinations")
        print(f"      Found: {formations_found} UNIQUE formations (one per Peak A)")
        
        return formations



def convert_formations_to_patterns(self, df, formations):
    """Convert A-B-C formations to full A-B-C-D-E patterns."""
    
    patterns = []
    print(f"\n🔄 CONVERTING {len(formations)} FORMATIONS TO FULL PATTERNS:")
    
    for i, formation in enumerate(formations):
        print(f"   📋 Converting formation {i+1}: {formation['peak_a_time'].strftime('%m-%d %H:%M')} → {formation['peak_c_time'].strftime('%m-%d %H:%M')}")
        
        # Find handle after Peak C
        handles = self.detect_handle_formation(
            df, 
            formation['natural_resistance'], 
            formation['peak_c_time'], 
            formation['peak_c_price']
        )
        
        if not handles:
            print(f"      ❌ No handle found")
            continue
        
        best_handle = max(handles, key=lambda h: h['score'])
        
        # Find breakout after handle
        breakout_time = self.find_breakout_after_handle(
            df, 
            best_handle['end'], 
            formation['natural_resistance'],
            formation['peak_a_price'],
            formation['peak_c_price'], 
            formation['peak_c_time']
        )

        if breakout_time:
            print(f"DEBUG: breakout_time={breakout_time}, handle_end={best_handle['end']}, peak_c={formation['peak_c_time']}")
            if breakout_time <= best_handle['end']:
                print(f"ERROR: Breakout {breakout_time} is not after handle {best_handle['end']}")
        
        if not breakout_time:
            print(f"      ❌ No breakout found")
            continue
        
        # Create full pattern
        pattern = {
            'peak_a': formation['peak_a_time'],
            'trough_b': formation['trough_b_time'],
            'peak_c': formation['peak_c_time'],
            'handle_d': best_handle['end'],  # Handle start time
            'breakout_e': breakout_time,
            'breakout_threshold': formation['natural_resistance'],
            'breakout_confirmed': True,
            'cup_depth': formation['peak_a_price'] - formation['trough_b_price'],
            'cup_depth_pct': ((formation['peak_a_price'] - formation['trough_b_price']) / formation['peak_a_price']) * 100,
            'handle_depth_pct': best_handle['depth_pct'],
            'symmetry_score': formation['symmetry_score'],
            'detection_method': 'formation_first'
        }
        
        patterns.append(pattern)
        print(f"      ✅ Pattern created: Handle {best_handle['depth_pct']:.2f}%, Breakout at {breakout_time}")
    
    print(f"\n📊 CONVERSION RESULTS: {len(patterns)} full patterns created from {len(formations)} formations")
    return patterns


def is_accumulation_period(self, period_data):
        """Check if period shows accumulation characteristics - RELAXED for 4H ES"""
    
        # RELAXED: Higher volatility tolerance for ES futures
        volatility = period_data['close'].std() / period_data['close'].mean()
        if volatility > 0.12:  # Increased from 5% to 12% for ES futures
            return False
        
        # RELAXED: Allow larger single moves for 4H timeframe
        max_move = abs(period_data['close'].pct_change()).max()
        if max_move > 0.15:  # Increased from 8% to 15% for 4H gaps
            return False
        
        # RELAXED: Less strict volatility compression requirement
        if len(period_data) >= 6:  # Only check if we have enough data
            first_half = period_data[:len(period_data)//2]
            second_half = period_data[len(period_data)//2:]
            
            first_vol = first_half['close'].std()
            second_vol = second_half['close'].std()
            
            # Allow volatility to increase up to 50% (was 20%)
            return second_vol <= first_vol * 1.5
        
        return True  # Accept shorter periods


def is_accumulation_period_detailed(self, period_data):
        """Check accumulation characteristics with detailed logging"""
        
        # RELAXED: Higher volatility tolerance for ES futures
        volatility = period_data['close'].std() / period_data['close'].mean()
        print(f"            📊 Volatility: {volatility:.4f} (limit: 0.12)")
        if volatility > 0.12:
            return False, f"High volatility: {volatility:.4f} > 0.12"
        
        # RELAXED: Allow larger single moves for 4H timeframe
        price_changes = period_data['close'].pct_change().abs()
        max_move = price_changes.max()
        print(f"            📈 Max single move: {max_move:.4f} (limit: 0.15)")
        if max_move > 0.15:
            return False, f"Large single move: {max_move:.4f} > 0.15"
        
        # RELAXED: Less strict volatility compression requirement
        if len(period_data) >= 6:
            first_half = period_data[:len(period_data)//2]
            second_half = period_data[len(period_data)//2:]
            
            first_vol = first_half['close'].std()
            second_vol = second_half['close'].std()
            
            vol_ratio = second_vol / (first_vol + 1e-10)  # Avoid division by zero
            print(f"            📉 Volatility compression: {vol_ratio:.2f} (limit: 1.5)")
            
            # Allow volatility to increase up to 50% (was 20%)
            if second_vol > first_vol * 1.5:
                return False, f"Volatility increased too much: {vol_ratio:.2f} > 1.5"
        
        return True, f"Valid: vol={volatility:.4f}, max_move={max_move:.4f}"


def calculate_enhanced_cup_roundness(self, df, cup_start, cup_bottom, cup_end):
        """Enhanced U-shape vs V-shape detection"""
        # Current V8 method + these additions:
        
        # Add: Bottom accumulation time analysis
        cup_data = df.loc[cup_start:cup_end]
        bottom_20pct = cup_data['low'].min() + (cup_data['high'].max() - cup_data['low'].min()) * 0.2
        time_in_bottom = len(cup_data[cup_data['low'] <= bottom_20pct]) / len(cup_data)
        
        # U-shapes spend 30%+ time in bottom 20% of cup
        if time_in_bottom < 0.3:
            return 0.2  # Likely V-shape
        
        # Add: Descent/ascent symmetry check
        mid_point = len(cup_data) // 2
        left_slope = abs(cup_data.iloc[0]['close'] - cup_data.iloc[mid_point]['close']) / mid_point
        right_slope = abs(cup_data.iloc[-1]['close'] - cup_data.iloc[mid_point]['close']) / (len(cup_data) - mid_point)
        
        slope_symmetry = 1 - abs(left_slope - right_slope) / max(left_slope, right_slope)
        
        return min(1.0, time_in_bottom * 2 + slope_symmetry * 0.5)


 
def validate_cup_roundness_professional(self, df, cup_start, cup_bottom, cup_end, price_col='close_smooth'):
        """Professional cup roundness validation - rejects V-shaped reversals."""
        try:
            # Extract cup segment
            cup_segment = df.loc[cup_start:cup_end]
            if len(cup_segment) < 10:  # Need minimum bars for meaningful analysis
                return 0.0, "Cup too short for analysis"
            
            # 1. SMOOTHED PRICE ANALYSIS
            # Use 3-bar SMA to extract true shape
            cup_prices = cup_segment['close'].rolling(window=3, center=True).mean().fillna(cup_segment['close'])
            
            # 2. BOTTOM ACCUMULATION CHECK
            cup_depth = cup_segment['high'].max() - cup_segment['low'].min()
            bottom_threshold = cup_segment['low'].min() + (cup_depth * 0.15)  # Bottom 15%
            
            # Count bars spent in bottom 15% of cup
            bottom_bars = len(cup_prices[cup_segment['low'] <= bottom_threshold])
            total_bars = len(cup_segment)
            bottom_ratio = bottom_bars / total_bars
            
            # REJECT if less than 20% of time spent near bottom (V-shape indicator)
            if bottom_ratio < 0.2:
                return 0.0, f"V-shaped: Only {bottom_ratio:.1%} time at bottom"
            
            # 3. SMOOTHNESS CHECK - Detect sharp reversals
            price_changes = cup_prices.diff().abs()
            volatility = price_changes.std()
            max_change = price_changes.max()
            
            # REJECT if single large price movement (spike reversal)
            if max_change > volatility * 3:
                return 0.0, f"Sharp reversal detected: {max_change:.2f} vs avg {volatility:.2f}"
            
            # 4. CURVATURE ANALYSIS - Fit polynomial
            x = np.linspace(0, 1, len(cup_prices))
            y = cup_prices.values
            
            # Normalize prices
            y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
            
            # Fit quadratic (U-shape should have positive x² coefficient)
            coeffs = np.polyfit(x, y_norm, 2)
            a, b, c = coeffs
            
            # Calculate curvature score
            if a <= 0:  # Negative or zero curvature = not a proper cup
                return 0.0, f"Inverted curvature: a={a:.3f}"
            
            # 5. SYMMETRY CHECK (enhanced)
            mid_point = len(cup_prices) // 2
            left_half = cup_prices.iloc[:mid_point]
            right_half = cup_prices.iloc[mid_point:].iloc[::-1]  # Reverse right side
            
            # Align lengths
            min_len = min(len(left_half), len(right_half))
            left_aligned = left_half.iloc[-min_len:].values
            right_aligned = right_half.iloc[:min_len].values
            
            # Calculate correlation (symmetry measure)
            if len(left_aligned) > 1:
                correlation = np.corrcoef(left_aligned, right_aligned)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
            
            # 6. COMPOSITE ROUNDNESS SCORE
            curvature_score = min(1.0, a * 2)  # Scale curvature coefficient
            bottom_score = min(1.0, bottom_ratio * 5)  # Reward time at bottom
            smoothness_score = min(1.0, 1 - (max_change / (volatility * 5)))  # Penalize spikes
            symmetry_score = max(0.0, correlation)
            
            # Weighted composite score
            roundness_score = (
                0.3 * curvature_score +
                0.3 * bottom_score +
                0.2 * smoothness_score +
                0.2 * symmetry_score
            )
            
            details = (f"Curvature: {curvature_score:.2f}, Bottom: {bottom_score:.2f}, "
                    f"Smooth: {smoothness_score:.2f}, Symmetry: {symmetry_score:.2f}")
            
            return roundness_score, details
            
        except Exception as e:
            return 0.0, f"Error: {str(e)}"
        



def detect_extrema_enhanced(self, df, price_col='close'):
        """Enhanced extrema detection with multiple validation methods"""
        
        # Method 1: Traditional peak/trough detection
        traditional_extrema = self.detect_extrema_multi_scale(df, f"{price_col}_smooth")
        
        # Method 2: Fractal-based detection
        fractal_extrema = self.detect_fractal_extrema(df)
        
        # Method 3: Combine all methods with confidence scoring
        combined_extrema = self.combine_extrema_methods(traditional_extrema, fractal_extrema)
        
        return combined_extrema
    

def detect_fractal_extrema(self, df, lookback=5):
        """Detect fractals (Williams %R style)"""
        extrema = np.zeros(len(df))
        
        for i in range(lookback, len(df) - lookback):
            # Fractal High: current high > all highs in lookback window
            window_highs = df['high'].iloc[i-lookback:i+lookback+1]
            if df['high'].iloc[i] == window_highs.max():
                extrema[i] = 1
                
            # Fractal Low: current low < all lows in lookback window  
            window_lows = df['low'].iloc[i-lookback:i+lookback+1]
            if df['low'].iloc[i] == window_lows.min():
                extrema[i] = -1
                
        return extrema


 
def combine_extrema_methods(self, traditional, fractals, volume_confirmed=None):
        """Combine multiple extrema detection methods"""
        combined = np.zeros(len(traditional))
        
        for i in range(len(combined)):
            votes_peak = 0
            votes_trough = 0
            
            if i < len(traditional) and traditional[i] == 1:
                votes_peak += 1
            if i < len(fractals) and fractals[i] == 1:
                votes_peak += 1
                
            if i < len(traditional) and traditional[i] == -1:
                votes_trough += 1
            if i < len(fractals) and fractals[i] == -1:
                votes_trough += 1
            
            # Need at least 1 vote for detection
            if votes_peak >= 1:
                combined[i] = 1
            elif votes_trough >= 1:
                combined[i] = -1
                
        return combined