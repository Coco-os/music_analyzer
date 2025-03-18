import streamlit as st
import os
import tempfile
import matplotlib.pyplot as plt
from music_analyzer import MusicAnalyzer
import numpy as np
import base64
from io import BytesIO

st.set_page_config(
    page_title="Music Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Music Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Upload an audio file to analyze its musical characteristics with a focus on cultural elements.</p>", unsafe_allow_html=True)

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'figures' not in st.session_state:
    st.session_state.figures = {}

with st.sidebar:
    st.markdown("<h2 class='sub-header'>Configuration</h2>", unsafe_allow_html=True)
    
    microtonal_threshold = st.slider(
        "Microtonal Threshold", 
        min_value=0.1, 
        max_value=0.5, 
        value=0.4, 
        step=0.05,
        help="Threshold to identify microtonal notes (distance from standard semitone)"
    )
    
    magnitude_threshold = st.slider(
        "Magnitude Threshold Percentile", 
        min_value=50, 
        max_value=95, 
        value=75, 
        step=5,
        help="Only consider notes above this percentile of magnitude"
    )
    
    melisma_length = st.slider(
        "Min Melisma Length", 
        min_value=3, 
        max_value=10, 
        value=5, 
        step=1,
        help="Minimum number of notes to consider a passage as melismatic"
    )
    
    st.markdown("### Scale Detection")
    scales_to_use = st.multiselect(
        "Scales to Include",
        options=[
            "Major", "Minor", "Pentatonic Major", "Pentatonic Minor", 
            "Blues", "Harmonic Minor", "Melodic Minor", "Dorian", 
            "Phrygian", "Lydian", "Mixolydian", "Locrian",
            "Berber (North African)"
        ],
        default=[
            "Major", "Minor", "Pentatonic Major", "Pentatonic Minor", 
            "Blues", "Harmonic Minor", "Dorian", "Phrygian"
        ]
    )
    
    st.markdown("### Visualization")
    cmap_spectrogram = st.selectbox(
        "Spectrogram Colormap",
        options=["magma", "viridis", "plasma", "inferno", "cividis", "mako"],
        index=0
    )
    
    cmap_chroma = st.selectbox(
        "Chromagram Colormap",
        options=["viridis", "plasma", "magma", "inferno", "cividis", "mako"],
        index=0
    )
    
    cmap_microtonal = st.selectbox(
        "Microtonal Colormap",
        options=["coolwarm", "RdBu_r", "seismic", "bwr"],
        index=0
    )

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "flac", "ogg", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    custom_config = {
        'analysis': {
            'microtonal_threshold': microtonal_threshold,
            'magnitude_threshold_percentile': magnitude_threshold,
            'min_melisma_length': melisma_length,
        },
        'visualization': {
            'cmap_spectrogram': cmap_spectrogram,
            'cmap_chroma': cmap_chroma,
            'cmap_microtonal': cmap_microtonal,
        }
    }
    
    default_scales = MusicAnalyzer.DEFAULT_CONFIG['scales']
    custom_config['scales'] = {k: v for k, v in default_scales.items() if k in scales_to_use}
    
    analyzer = MusicAnalyzer()
    analyzer.config['analysis'].update(custom_config['analysis'])
    analyzer.config['visualization'].update(custom_config['visualization'])
    analyzer.config['scales'] = custom_config['scales']
    analyzer.scale_patterns = {
        name: set(pattern) 
        for name, pattern in analyzer.config['scales'].items()
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Loading audio file...")
    progress_bar.progress(10)
    
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    original_show = plt.show
    figures = {}
    
    def capture_show():
        fig = plt.gcf()
        fig_name = f"figure_{len(figures)}"
        figures[fig_name] = fig_to_base64(fig)
        plt.close(fig)
    
    plt.show = capture_show
    
    try:
        status_text.text("Loading audio...")
        y, sr = analyzer._load_audio(tmp_path)
        progress_bar.progress(20)
        
        status_text.text("Detecting microtonality...")
        result = analyzer.detect_microtonality(y, sr)

        microtonal_pitches = result["microtonal_pitches"]
        microtonal_times = result["times"]
        deviations = result["deviations"]
        glissando_count = result["glissando_count"]
        vibrato_count = result["vibrato_count"]
        intentional_score = result["intentional_microtonality_score"]
        progress_bar.progress(40)
        
        status_text.text("Detecting scale...")
        scale_name, tonic, chroma = analyzer.detect_scale(y, sr)
        progress_bar.progress(60)
        
        status_text.text("Analyzing melisma and rhythm...")
        melisma_count, pitch_contour = analyzer.detect_melisma(y, sr)
        call_response, onset_times, onset_env = analyzer.detect_call_response(y, sr)
        tempo, complex_rhythm, beat_times, beat_intervals = analyzer.detect_complex_rhythm(y, sr)
        progress_bar.progress(80)
        
        # Compile results
        file_stem = os.path.splitext(os.path.basename(uploaded_file.name))[0]
        
        analysis_results = {
            'filename': uploaded_file.name,
            'duration': len(y) / sr,
            'sample_rate': sr,
            'microtonal_pitches': microtonal_pitches,
            'microtonal_times': microtonal_times, 
            'deviations': deviations,
            'microtonality_score': len(microtonal_pitches) / max(1, len(y) / sr * 10),
            'intentional microtonality score': glissando_count,
            'glissando': glissando_count,
            'vibrato' : vibrato_count,
            'scale': scale_name,
            'tonic': tonic,
            'tonic_note': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][tonic],
            'chroma': chroma,
            'melisma_count': melisma_count,
            'melisma_score': min(1.0, melisma_count / 10),
            'pitch_contour': pitch_contour,
            'call_response': call_response,
            'tempo': tempo,
            'complex_rhythm': complex_rhythm,
            'beat_times': beat_times,
            'onset_times': onset_times,
            'onset_env': onset_env,
            'rhythm_complexity_score': 0.2 + (0.8 * int(complex_rhythm)),
        }
        
        # Calculate overall cultural characteristics score
        cultural_features = [
            ('Microtonality', analysis_results['microtonality_score']),
            ('Melismatic Singing', analysis_results['melisma_score']),
            ('Call and Response', float(call_response)),
            ('Rhythmic Complexity', analysis_results['rhythm_complexity_score'])
        ]
        analysis_results['cultural_features'] = cultural_features
        
        # Generate visualizations
        status_text.text("Generating visualizations...")
        analyzer.create_feature_visualization(analysis_results, file_stem, ".")
        analyzer.create_spectrogram_visualization(y, sr, file_stem, ".")
        progress_bar.progress(100)
        
        # Store results in session state
        st.session_state.results = analysis_results
        st.session_state.figures = figures
        st.session_state.analysis_complete = True
        
        # Clean up
        os.unlink(tmp_path)
        
        status_text.text("Analysis complete!")
        
    except Exception as e:
        st.error(f"Error analyzing audio: {str(e)}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    finally:
        # Restore original plt.show
        plt.show = original_show

# Display results if analysis is complete
if st.session_state.analysis_complete and st.session_state.results is not None:
    results = st.session_state.results
    
    st.markdown("<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
    
    # Create tabs for different result sections
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Scale & Pitch", "Rhythm", "Visualizations"])
    
    with tab1:
        st.markdown("### Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Duration", f"{results['duration']:.2f} seconds")
            st.metric("Sample Rate", f"{results['sample_rate']} Hz")
            st.metric("Tempo", f"{results['tempo']:.1f} BPM")
        
        with col2:
            st.metric("Scale", results['scale'])
            st.metric("Tonic", results['tonic_note'])
            st.metric("Melismatic Phrases", results['melisma_count'])
        
        st.markdown("### Cultural Features")
        
        for feature, score in results['cultural_features']:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{feature}:**")
            with col2:
                st.progress(float(score))
                st.write(f"{score:.3f}")
    
    with tab2:
        st.markdown("### Scale and Pitch Analysis")
        
        # Display chromagram as a bar chart
        st.subheader("Note Distribution")
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chroma_data = results.get('chroma', np.zeros(12))
        
        # Create a bar chart
        fig, ax = plt.figure(figsize=(10, 4)), plt.gca()
        bars = ax.bar(note_names, chroma_data, color='skyblue')
        ax.axvline(x=note_names[results['tonic']], color='red', linestyle='--', alpha=0.7)
        ax.set_ylabel('Prominence')
        ax.set_title(f"Detected Scale: {results['scale']} (Tonic: {results['tonic_note']})")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Microtonal information
        st.subheader("Microtonality")
        microtonal_count = len(results.get('microtonal_pitches', []))
        if microtonal_count > 0:
            st.write(f"Detected {microtonal_count} microtonal notes")
            st.write(f"Microtonality Score: {results['microtonality_score']:.3f}")
        else:
            st.write("No significant microtonal elements detected")
    
    with tab3:
        st.markdown("### Rhythm Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Tempo", f"{results['tempo']:.1f} BPM")
            st.metric("Complex Rhythm", "Yes" if results['complex_rhythm'] else "No")
        
        with col2:
            st.metric("Call and Response", "Detected" if results['call_response'] else "Not Detected")
            st.metric("Rhythm Complexity Score", f"{results['rhythm_complexity_score']:.3f}")
    
    with tab4:
        st.markdown("### Visualizations")
        
        for fig_name, fig_data in st.session_state.figures.items():
            st.image(f"data:image/png;base64,{fig_data}", use_column_width=True)

with st.expander("About Music Analyzer"):
    st.write("""
    The Music Analyzer is a tool for analyzing musical characteristics in audio files with a focus on cultural elements.
    
    It extracts and visualizes features such as:
    - Microtonality (notes between the standard 12-tone equal temperament)
    - Scale detection (identifying the musical scale and tonic note)
    - Melismatic singing (ornamental vocal runs)
    - Call and response patterns
    - Rhythmic complexity
    
    This tool can be useful for ethnomusicologists, music researchers, composers, and anyone interested in analyzing
    the cultural characteristics of music.
    """)
    
    st.markdown("### How to Use")
    st.write("""
    1. Upload an audio file using the file uploader
    2. Adjust analysis parameters in the sidebar if needed
    3. Wait for the analysis to complete
    4. Explore the results in the different tabs
    """)
