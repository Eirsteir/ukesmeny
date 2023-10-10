from contextlib import contextmanager

from sqlalchemy import create_engine, Column, String, Text, ForeignKey, Table
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, sessionmaker
import uuid

Base = declarative_base()

# Association table for the many-to-many relationship between Recipe and Ingredient
recipe_ingredient_association = Table(
    "recipe_ingredient",
    Base.metadata,
    Column("recipe_id", UUID(as_uuid=True), ForeignKey("recipe.id"), primary_key=True),
    Column(
        "ingredient_id",
        UUID(as_uuid=True),
        ForeignKey("ingredient.id"),
        primary_key=True,
    ),
)


class Recipe(Base):
    __tablename__ = "recipe"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False,
    )
    name = Column(String, nullable=False)
    description = Column(String(512))
    ingredients = relationship(
        "Ingredient", secondary=recipe_ingredient_association, back_populates="recipes"
    )
    source_id = Column


class Ingredient(Base):
    __tablename__ = "ingredient"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False,
    )
    title = Column(String, nullable=False)
    recipes = relationship(
        "Recipe", secondary=recipe_ingredient_association, back_populates="ingredients"
    )


@contextmanager
def get_session():
    engine = create_engine("postgresql://admin:password@localhost:5432/db")

    # Create a session factory
    Session = sessionmaker(bind=engine)

    # Create a new session instance
    session = Session()

    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


# Example usage
if __name__ == "__main__":
    engine = create_engine("postgresql://admin:password@localhost:5432/db")
    Base.metadata.create_all(engine)
