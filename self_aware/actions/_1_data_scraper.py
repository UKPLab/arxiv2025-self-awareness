import json
import logging
import os
import time
from pathlib import Path

from aim import Run
from omegaconf import DictConfig
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm
from urartu.common.action import ActionDataset
from self_aware.utils.utils import set_random_seeds


class DataScraper(ActionDataset):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def initialize(self):
        set_random_seeds(self.action_cfg.seed)
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.sparql.agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"

    def query_player(self, num_entities, max_num_relations, batch_size):
        start_time = time.time()
        entity_query = """
            SELECT DISTINCT ?player
            WHERE {{
                ?player wdt:P106 wd:Q937857;  # instance of athlete
                    wdt:P641 ?sport.
            }}
            LIMIT {num_entities}
        """.format(
            num_entities=num_entities
        )

        self.sparql.setQuery(entity_query)
        self.sparql.setReturnFormat(JSON)
        entities_results = self.sparql.query().convert()
        players = [
            result["player"]["value"]
            for result in entities_results["results"]["bindings"]
        ]

        end_time = time.time()
        logging.info(f"Entity query took {end_time - start_time:.2f} seconds")

        all_results = []
        player_batches = [
            players[i : i + batch_size] for i in range(0, len(players), batch_size)
        ]

        start_time = time.time()
        for batch in tqdm(player_batches):
            batch_query = """
                SELECT DISTINCT ?player ?playerLabel ?birthplaceLabel ?birthdate ?positionLabel ?nationalityLabel
                WHERE {{
                    VALUES ?player {{ {players} }}
                    OPTIONAL {{ ?player wdt:P19 ?birthplace. }}  # birthplace
                    OPTIONAL {{ ?player wdt:P569 ?birthdate. }}  # birthdate
                    OPTIONAL {{ ?player wdt:P413 ?position. }}    # position played
                    OPTIONAL {{ ?player wdt:P27 ?nationality. }} # nationality
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }}
                }}
                LIMIT {max_num_relations}
            """.format(
                players=" ".join(f"<{player}>" for player in batch),
                max_num_relations=max_num_relations * len(batch),
            )

            self.sparql.setQuery(batch_query)
            self.sparql.setReturnFormat(JSON)
            relation_results = self.sparql.query().convert()
            all_results.extend(relation_results["results"]["bindings"])

        end_time = time.time()
        logging.info(f"Relation query took {end_time - start_time:.2f} seconds")

        return all_results

    def query_movie(self, num_entities, max_num_relations, batch_size):
        start_time = time.time()
        entity_query = """
            SELECT DISTINCT ?movie
            WHERE {{
                ?movie wdt:P31 wd:Q11424.  # instance of film
            }}
            LIMIT {num_entities}
        """.format(
            num_entities=num_entities
        )

        self.sparql.setQuery(entity_query)
        self.sparql.setReturnFormat(JSON)
        entities_results = self.sparql.query().convert()
        movies = [
            result["movie"]["value"]
            for result in entities_results["results"]["bindings"]
        ]

        end_time = time.time()
        logging.info(f"Entity query took {end_time - start_time:.2f} seconds")

        all_results = []
        movie_batches = [
            movies[i : i + batch_size] for i in range(0, len(movies), batch_size)
        ]
        start_time = time.time()
        for batch in tqdm(movie_batches):
            batch_query = """
                SELECT DISTINCT ?movie ?movieLabel ?directorLabel ?releaseDate ?genreLabel ?countryLabel
                WHERE {{
                    VALUES ?movie {{ {movies} }}
                    OPTIONAL {{ ?movie wdt:P57 ?director. }}       # director
                    OPTIONAL {{ ?movie wdt:P577 ?releaseDate. }}   # release date
                    OPTIONAL {{ ?movie wdt:P136 ?genre. }}         # genre
                    OPTIONAL {{ ?movie wdt:P495 ?country. }}       # country of origin
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                }}
                LIMIT {max_num_relations}
            """.format(
                movies=" ".join(f"<{movie}>" for movie in batch),
                max_num_relations=max_num_relations * len(batch),
            )

            self.sparql.setQuery(batch_query)
            self.sparql.setReturnFormat(JSON)
            relation_results = self.sparql.query().convert()
            all_results.extend(relation_results["results"]["bindings"])

        end_time = time.time()
        logging.info(f"Relation query took {end_time - start_time:.2f} seconds")

        return all_results

    def query_city(self, num_entities, max_num_relations, batch_size):
        start_time = time.time()
        entity_query = """
            SELECT DISTINCT ?city
            WHERE {{
                ?city wdt:P31/wdt:P279* wd:Q515.  # instance of city
            }}
            LIMIT {num_entities}
        """.format(
            num_entities=num_entities
        )

        self.sparql.setQuery(entity_query)
        self.sparql.setReturnFormat(JSON)
        entities_results = self.sparql.query().convert()
        cities = [
            result["city"]["value"]
            for result in entities_results["results"]["bindings"]
        ]

        end_time = time.time()
        logging.info(f"Entity query took {end_time - start_time:.2f} seconds")

        all_results = []
        city_batches = [
            cities[i : i + batch_size] for i in range(0, len(cities), batch_size)
        ]
        start_time = time.time()
        for batch in tqdm(city_batches):
            batch_query = """
                SELECT DISTINCT ?city ?cityLabel ?countryLabel ?mayorLabel ?foundedDate ?climateLabel
                WHERE {{
                    VALUES ?city {{ {cities} }}
                    OPTIONAL {{ ?city wdt:P17 ?country. }}        # country
                    OPTIONAL {{ ?city wdt:P6 ?mayor. }}           # head of government (mayor)
                    OPTIONAL {{ ?city wdt:P571 ?foundedDate. }}   # founded date
                    OPTIONAL {{ ?city wdt:P2564 ?climate. }}      # climate (Köppen classification)
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                }}
                LIMIT {max_num_relations}
            """.format(
                cities=" ".join(f"<{city}>" for city in batch),
                max_num_relations=max_num_relations * len(batch),
            )

            self.sparql.setQuery(batch_query)
            self.sparql.setReturnFormat(JSON)
            relation_results = self.sparql.query().convert()
            all_results.extend(relation_results["results"]["bindings"])

        end_time = time.time()
        logging.info(f"Relation query took {end_time - start_time:.2f} seconds")

        return all_results

    def query_song(self, num_entities, max_num_relations, batch_size):
        start_time = time.time()
        entity_query = """
            SELECT DISTINCT ?song
            WHERE {{
                ?song wdt:P31 wd:Q7366.  # instance of song
            }}
            LIMIT {num_entities}
        """.format(
            num_entities=num_entities
        )

        self.sparql.setQuery(entity_query)
        self.sparql.setReturnFormat(JSON)
        entities_results = self.sparql.query().convert()
        songs = [
            result["song"]["value"]
            for result in entities_results["results"]["bindings"]
        ]

        end_time = time.time()
        logging.info(f"Entity query took {end_time - start_time:.2f} seconds")

        all_results = []
        song_batches = [
            songs[i : i + batch_size] for i in range(0, len(songs), batch_size)
        ]
        start_time = time.time()
        for batch in tqdm(song_batches):
            batch_query = """
                SELECT DISTINCT ?song ?songLabel ?artistLabel ?albumLabel ?releaseDate ?languageLabel
                WHERE {{
                    VALUES ?song {{ {songs} }}
                    OPTIONAL {{ ?song wdt:P175 ?artist. }}         # artist
                    OPTIONAL {{ ?song wdt:P361 ?album. }}          # part of an album
                    OPTIONAL {{ ?song wdt:P577 ?releaseDate. }}    # release date
                    OPTIONAL {{ ?song wdt:P407 ?language. }}       # language
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                }}
                LIMIT {max_num_relations}
            """.format(
                songs=" ".join(f"<{song}>" for song in batch),
                max_num_relations=max_num_relations * len(batch),
            )

            self.sparql.setQuery(batch_query)
            self.sparql.setReturnFormat(JSON)
            relation_results = self.sparql.query().convert()
            all_results.extend(relation_results["results"]["bindings"])

        end_time = time.time()
        logging.info(f"Relation query took {end_time - start_time:.2f} seconds")

        return all_results

    def run(self):
        run_dir = Path(self.cfg.run_dir)
        entries_dir = run_dir.joinpath("wikidata_entities")
        os.makedirs(entries_dir, exist_ok=True)

        for entity_type in tqdm(self.task_cfg.dataset.entity_types):
            entity_query = getattr(self, f"query_{entity_type}")
            data = entity_query(
                num_entities=self.task_cfg.dataset.num_entities,
                max_num_relations=self.task_cfg.dataset.max_num_relations,
                batch_size=self.task_cfg.dataset.batch_size,
            )
            with open(
                entries_dir.joinpath(f"{entity_type}.jsonl"), "w", encoding="utf-8"
            ) as f:
                for item in data:
                    item["entity_type"] = entity_type
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(cfg: DictConfig, aim_run: Run):
    data_scraper = DataScraper(cfg, aim_run)
    data_scraper.initialize()
    data_scraper.run()
