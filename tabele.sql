CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL,
    title TEXT,
    tags TEXT[],
    project_id TEXT,
    experiment_id TEXT,
    capture_time TIMESTAMP,
    camera_model TEXT,
    iso INT,
    aperture REAL,
    focal_length REAL,
    gps_lat DOUBLE PRECISION,
    gps_lng DOUBLE PRECISION,
    text_embedding VECTOR(768)  -- kolumna pgvector
);

CREATE TABLE notes (
    id SERIAL PRIMARY KEY,
    experiment_id TEXT,
    language TEXT,
    text TEXT
);

CREATE TABLE image_notes (
    image_id INT REFERENCES images(id),
    note_id INT REFERENCES notes(id),
    PRIMARY KEY (image_id, note_id)
);
