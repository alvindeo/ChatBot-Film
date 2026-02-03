# -*- coding: utf-8 -*-
"""
Movie Dataset Loader
Mengintegrasikan dataset film publik dari berbagai sumber
"""

import os
import json
import pandas as pd
import requests
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm


class MovieDataset:
    """
    Loader untuk dataset film publik
    Mendukung: TMDb, IMDb, dan custom datasets
    """
    
    def __init__(self, cache_dir='dataset/cache'):
        """
        Initialize dataset loader
        
        Args:
            cache_dir: Directory untuk menyimpan cached datasets
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.movies = []
        
    def load_tmdb_popular(self, api_key: Optional[str] = None, pages: int = 5) -> List[Dict]:
        """
        Load popular movies dari TMDb API
        
        Args:
            api_key: TMDb API key (dapatkan di https://www.themoviedb.org/settings/api)
            pages: Jumlah halaman yang akan di-load (setiap halaman = 20 film)
        
        Returns:
            List of movie dictionaries
        """
        if not api_key:
            api_key = os.getenv('TMDB_API_KEY')
        
        if not api_key or api_key == 'your_tmdb_api_key_here':
            print("⚠️ TMDb API key tidak tersedia. Skip loading dari TMDb.")
            return []
        
        movies = []
        base_url = "https://api.themoviedb.org/3/movie/popular"
        
        try:
            for page in range(1, pages + 1):
                response = requests.get(
                    base_url,
                    params={'api_key': api_key, 'language': 'id-ID', 'page': page}
                )
                response.raise_for_status()
                data = response.json()
                
                for movie in data.get('results', []):
                    movies.append({
                        'title': movie.get('title', ''),
                        'original_title': movie.get('original_title', ''),
                        'overview': movie.get('overview', ''),
                        'release_date': movie.get('release_date', ''),
                        'vote_average': movie.get('vote_average', 0),
                        'vote_count': movie.get('vote_count', 0),
                        'popularity': movie.get('popularity', 0),
                        'genre_ids': movie.get('genre_ids', []),
                        'source': 'tmdb'
                    })
                
                print(f"✅ Loaded page {page}/{pages} from TMDb ({len(movies)} movies total)")
            
            # Save to cache
            cache_file = os.path.join(self.cache_dir, 'tmdb_popular.json')
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(movies, f, ensure_ascii=False, indent=2)
            
            return movies
            
        except Exception as e:
            print(f"⚠️ Error loading from TMDb: {e}")
            return []
    
    def load_sample_dataset(self) -> List[Dict]:
        """
        Load sample dataset film populer (built-in)
        Dataset ini sudah include tanpa perlu API key
        """
        sample_movies = [
            {
                'title': 'The Shawshank Redemption',
                'year': 1994,
                'director': 'Frank Darabont',
                'genre': 'Drama',
                'rating': 9.3,
                'overview': 'Dua pria yang dipenjara menjalin ikatan selama bertahun-tahun, menemukan penghiburan dan penebusan melalui tindakan kebaikan bersama.',
                'cast': ['Tim Robbins', 'Morgan Freeman'],
                'source': 'sample'
            },
            {
                'title': 'The Godfather',
                'year': 1972,
                'director': 'Francis Ford Coppola',
                'genre': 'Crime, Drama',
                'rating': 9.2,
                'overview': 'Kisah tentang keluarga mafia Corleone dan bagaimana putra bungsu yang enggan mengambil alih bisnis keluarga.',
                'cast': ['Marlon Brando', 'Al Pacino', 'James Caan'],
                'source': 'sample'
            },
            {
                'title': 'The Dark Knight',
                'year': 2008,
                'director': 'Christopher Nolan',
                'genre': 'Action, Crime, Drama',
                'rating': 9.0,
                'overview': 'Batman menghadapi musuh terbesarnya, Joker, yang menciptakan kekacauan di Gotham City.',
                'cast': ['Christian Bale', 'Heath Ledger', 'Aaron Eckhart'],
                'source': 'sample'
            },
            {
                'title': 'Inception',
                'year': 2010,
                'director': 'Christopher Nolan',
                'genre': 'Action, Sci-Fi, Thriller',
                'rating': 8.8,
                'overview': 'Seorang pencuri yang mencuri rahasia perusahaan melalui teknologi berbagi mimpi diberi tugas terbalik: menanamkan ide ke dalam pikiran CEO.',
                'cast': ['Leonardo DiCaprio', 'Joseph Gordon-Levitt', 'Ellen Page'],
                'source': 'sample'
            },
            {
                'title': 'Pulp Fiction',
                'year': 1994,
                'director': 'Quentin Tarantino',
                'genre': 'Crime, Drama',
                'rating': 8.9,
                'overview': 'Kehidupan dua pembunuh bayaran, seorang petinju, istri gangster, dan sepasang perampok restoran saling terkait dalam empat cerita kekerasan dan penebusan.',
                'cast': ['John Travolta', 'Uma Thurman', 'Samuel L. Jackson'],
                'source': 'sample'
            },
            {
                'title': 'Forrest Gump',
                'year': 1994,
                'director': 'Robert Zemeckis',
                'genre': 'Drama, Romance',
                'rating': 8.8,
                'overview': 'Kisah hidup Forrest Gump, seorang pria dengan IQ rendah yang tanpa disadari menjadi saksi dan berpengaruh pada beberapa peristiwa penting abad ke-20.',
                'cast': ['Tom Hanks', 'Robin Wright', 'Gary Sinise'],
                'source': 'sample'
            },
            {
                'title': 'Interstellar',
                'year': 2014,
                'director': 'Christopher Nolan',
                'genre': 'Adventure, Drama, Sci-Fi',
                'rating': 8.6,
                'overview': 'Tim penjelajah menggunakan lubang cacing yang baru ditemukan untuk melampaui batas perjalanan ruang angkasa manusia dan menaklukkan jarak luas yang terlibat dalam perjalanan antarbintang.',
                'cast': ['Matthew McConaughey', 'Anne Hathaway', 'Jessica Chastain'],
                'source': 'sample'
            },
            {
                'title': 'Parasite',
                'year': 2019,
                'director': 'Bong Joon-ho',
                'genre': 'Comedy, Drama, Thriller',
                'rating': 8.6,
                'overview': 'Keserakahan dan diskriminasi kelas mengancam hubungan simbiosis baru antara keluarga kaya Park dan klan Kim yang miskin.',
                'cast': ['Song Kang-ho', 'Lee Sun-kyun', 'Cho Yeo-jeong'],
                'source': 'sample'
            },
            {
                'title': 'The Matrix',
                'year': 1999,
                'director': 'Lana Wachowski, Lilly Wachowski',
                'genre': 'Action, Sci-Fi',
                'rating': 8.7,
                'overview': 'Seorang hacker komputer belajar dari pemberontak misterius tentang sifat sebenarnya dari realitasnya dan perannya dalam perang melawan pengontrolnya.',
                'cast': ['Keanu Reeves', 'Laurence Fishburne', 'Carrie-Anne Moss'],
                'source': 'sample'
            },
            {
                'title': 'Avengers: Endgame',
                'year': 2019,
                'director': 'Anthony Russo, Joe Russo',
                'genre': 'Action, Adventure, Sci-Fi',
                'rating': 8.4,
                'overview': 'Setelah peristiwa Infinity War, Avengers berkumpul sekali lagi untuk membalikkan tindakan Thanos dan mengembalikan keseimbangan alam semesta.',
                'cast': ['Robert Downey Jr.', 'Chris Evans', 'Scarlett Johansson'],
                'source': 'sample'
            },
            {
                'title': 'Pengabdi Setan 2: Communion',
                'year': 2022,
                'director': 'Joko Anwar',
                'genre': 'Horror, Mystery',
                'rating': 6.7,
                'overview': 'Keluarga yang dikejar oleh roh jahat pindah ke apartemen di Jakarta, tapi teror tidak berhenti di sana.',
                'cast': ['Tara Basro', 'Bront Palarae', 'Endy Arfian'],
                'source': 'sample'
            },
            {
                'title': 'Laskar Pelangi',
                'year': 2008,
                'director': 'Riri Riza',
                'genre': 'Drama, Family',
                'rating': 7.9,
                'overview': 'Kisah inspiratif tentang 10 anak dari keluarga miskin di Belitung yang berjuang untuk mendapatkan pendidikan.',
                'cast': ['Cut Mini', 'Ikranagara', 'Tora Sudiro'],
                'source': 'sample'
            },
            {
                'title': 'Dilan 1990',
                'year': 2018,
                'director': 'Fajar Bustomi',
                'genre': 'Drama, Romance',
                'rating': 7.5,
                'overview': 'Kisah cinta antara Milea dan Dilan, seorang siswa SMA yang romantis dan pemberani di Bandung tahun 1990.',
                'cast': ['Iqbaal Ramadhan', 'Vanesha Prescilla', 'Sissy Priscillia'],
                'source': 'sample'
            },
            {
                'title': 'Gundala',
                'year': 2019,
                'director': 'Joko Anwar',
                'genre': 'Action, Adventure, Sci-Fi',
                'rating': 6.1,
                'overview': 'Sancaka hidup di jalanan Jakarta. Suatu hari, ia mengembangkan kekuatan luar biasa dan harus menghadapi kejahatan yang mengancam kota.',
                'cast': ['Abimana Aryasatya', 'Tara Basro', 'Bront Palarae'],
                'source': 'sample'
            },
            {
                'title': 'Habibie & Ainun',
                'year': 2012,
                'director': 'Faozan Rizal',
                'genre': 'Biography, Drama, Romance',
                'rating': 7.6,
                'overview': 'Kisah cinta sejati antara B.J. Habibie dan istrinya Hasri Ainun Habibie.',
                'cast': ['Reza Rahadian', 'Bunga Citra Lestari', 'Tio Pakusadewo'],
                'source': 'sample'
            }
        ]
        
        # Save to cache
        cache_file = os.path.join(self.cache_dir, 'sample_movies.json')
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(sample_movies, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Loaded {len(sample_movies)} sample movies")
        return sample_movies
    
    def load_from_cache(self, source: str = 'sample') -> List[Dict]:
        """
        Load dataset dari cache
        
        Args:
            source: 'sample', 'tmdb', atau 'all'
        """
        movies = []
        
        if source in ['sample', 'all']:
            cache_file = os.path.join(self.cache_dir, 'sample_movies.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    movies.extend(json.load(f))
                print(f"✅ Loaded sample movies from cache")
        
        if source in ['tmdb', 'all']:
            cache_file = os.path.join(self.cache_dir, 'tmdb_popular.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    movies.extend(json.load(f))
                print(f"✅ Loaded TMDb movies from cache")
        
        return movies
    
    def convert_to_qa_format(self, movies: List[Dict]) -> tuple:
        """
        Convert movie dataset ke format Q&A untuk search engine
        
        Returns:
            tuple: (questions, answers, file_names)
        """
        questions = []
        answers = []
        file_names = []
        
        for movie in tqdm(movies, desc="🎞️ Converting movies to Q&A", unit="movie"):
            # Extract info
            title = movie.get('title', movie.get('original_title', 'Unknown'))
            year = movie.get('year', movie.get('release_date', '')[:4] if movie.get('release_date') else 'N/A')
            director = movie.get('director', 'N/A')
            genre = movie.get('genre', 'N/A')
            rating = movie.get('rating', movie.get('vote_average', 'N/A'))
            overview = movie.get('overview', 'Tidak ada deskripsi.')
            cast = movie.get('cast', [])
            source = movie.get('source', 'dataset')
            
            # Format cast
            cast_str = ', '.join(cast) if isinstance(cast, list) else str(cast)
            
            # Generate Q&A pairs
            
            # 1. Info umum film
            q1 = f"informasi film {title}"
            a1 = f"🎬 **{title}** ({year})\n\n📝 Sinopsis: {overview}\n\n⭐ Rating: {rating}\n🎭 Genre: {genre}\n🎬 Sutradara: {director}\n👥 Pemeran: {cast_str}"
            questions.append(q1)
            answers.append(a1)
            file_names.append(f"dataset_film_{source}.txt")
            
            # 2. Sutradara
            if director != 'N/A':
                q2 = f"siapa sutradara film {title}"
                a2 = f"Film **{title}** ({year}) disutradarai oleh **{director}**."
                questions.append(q2)
                answers.append(a2)
                file_names.append(f"dataset_film_{source}.txt")
            
            # 3. Genre
            if genre != 'N/A':
                q3 = f"genre film {title}"
                a3 = f"Film **{title}** bergenre **{genre}**."
                questions.append(q3)
                answers.append(a3)
                file_names.append(f"dataset_film_{source}.txt")
            
            # 4. Pemeran
            if cast_str:
                q4 = f"siapa pemeran film {title}"
                a4 = f"Film **{title}** dibintangi oleh: {cast_str}."
                questions.append(q4)
                answers.append(a4)
                file_names.append(f"dataset_film_{source}.txt")
            
            # 5. Tahun rilis
            if year != 'N/A':
                q5 = f"kapan film {title} rilis"
                a5 = f"Film **{title}** dirilis pada tahun **{year}**."
                questions.append(q5)
                answers.append(a5)
                file_names.append(f"dataset_film_{source}.txt")
        
        print(f"✅ Converted {len(movies)} movies to {len(questions)} Q&A pairs")
        return questions, answers, file_names
    
    def load_all(self, use_tmdb: bool = False, tmdb_api_key: Optional[str] = None) -> tuple:
        """
        Load semua dataset dan convert ke format Q&A
        
        Args:
            use_tmdb: Apakah menggunakan TMDb API
            tmdb_api_key: TMDb API key (optional)
        
        Returns:
            tuple: (questions, answers, file_names)
        """
        all_movies = []
        
        # Load sample dataset (always)
        sample_movies = self.load_sample_dataset()
        all_movies.extend(sample_movies)
        
        # Load TMDb if requested
        if use_tmdb and tmdb_api_key:
            tmdb_movies = self.load_tmdb_popular(api_key=tmdb_api_key, pages=3)
            all_movies.extend(tmdb_movies)
        
        # Convert to Q&A format
        return self.convert_to_qa_format(all_movies)


# === Testing ===
if __name__ == "__main__":
    # Fix encoding untuk Windows
    import sys
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("🧪 Testing Movie Dataset Loader\n")
    
    dataset = MovieDataset()
    
    # Load sample dataset
    print("=" * 50)
    print("Loading Sample Dataset")
    print("=" * 50)
    questions, answers, file_names = dataset.load_all(use_tmdb=False)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Total Q&A pairs: {len(questions)}")
    print(f"  - Unique sources: {len(set(file_names))}")
    
    print(f"\n📝 Sample Q&A:")
    for i in range(min(3, len(questions))):
        print(f"\nQ{i+1}: {questions[i]}")
        print(f"A{i+1}: {answers[i][:200]}...")
    
    print("\n💡 Tips:")
    print("1. Dataset sample sudah include 15 film populer")
    print("2. Untuk lebih banyak film, dapatkan TMDb API key (GRATIS): https://www.themoviedb.org/settings/api")
    print("3. Tambahkan TMDB_API_KEY ke file .env")
