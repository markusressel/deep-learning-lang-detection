# kiddEichlerToNcbi.sql was originally generated by the autoSql program, which also 
# generated kiddEichlerToNcbi.c and kiddEichlerToNcbi.h.  This creates the database representation of
# an object which can be loaded and saved from RAM in a fairly 
# automatic way.

#Map clone IDs from Kidd..Eichler '08 to NCBI clone and/or end read accessions
CREATE TABLE kiddEichlerToNcbi (
    name varchar(255) not null,	# Clone ID used in Kidd..Eichler '08
    cloneAcc varchar(255) not null,	# NCBI clone accession
    endF int not null,	# NCBI trace_name of forward end read
    endR int not null,	# NCBI trace_name of reverse end read
              #Indices
    PRIMARY KEY(name)
);
