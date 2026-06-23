"""
Classes for Tracewin files handling 
from ESS: https://gitlab.esss.lu.se/ess-bp/ess-python-tools/-/blob/mamad/ess/TraceWin.py
"""

from __future__ import print_function


class Dst:
    """
    Simple class to read in a
    TraceWin distribution file

    Class afterwards hold the following
    dictionary items:
      - x [m]
      - xp [rad]
      - y [m]
      - yp [rad]
      - phi [rad]
      - E [MeV] (kinetic energy)
    """

    def __init__(self, filename=None):
        # easy storage..
        self.filename = filename
        # used to create dict behaviour..
        self._columns = ["x", "xp", "y", "yp", "phi", "E"]
        if filename:
            # read in the file..
            self._readBinaryFile()
        else:
            import numpy

            self.Np = 0
            self.Ib = 0.0
            self.freq = 352.21
            self._data = numpy.zeros((self.Np, 6))
            self.mass = 0.0

    def append(self, x=0.0, xp=0.0, y=0.0, yp=0.0, E=0.0, phi=0.0):
        """
        Append one particle to the distribution

        - Kinetic Energy in MeV
        - x,y in m
        - xp,yp in rad
        - phi in rad
        """
        import numpy

        self._data = numpy.append(self._data, [[x, xp, y, yp, phi, E]], 0)
        self.Np += 1

    def remove(self, i=None):
        """
        Removes all particles from the distribution, or the line specified by i
        """
        import numpy

        if i is None:
            self._data = numpy.delete(self._data, numpy.s_[:], 0)
            self.Np = 0
        else:
            self._data = numpy.delete(self._data, i, 0)
            self.Np -= 1

    def _readBinaryFile(self):
        # Thanks Ema!

        import numpy

        fin = open(self.filename, "r")

        # dummy, Np, Ib, freq, dummy
        Header_type = numpy.dtype(
            [
                ("dummy12", numpy.int16),
                ("Np", numpy.int32),
                ("Ib", numpy.float64),
                ("freq", numpy.float64),
                ("dummy3", numpy.int8),
            ]
        )
        Header = numpy.fromfile(fin, dtype=Header_type, count=1)
        self.Np = Header["Np"][0]
        self.Ib = Header["Ib"][0]
        self.freq = Header["freq"][0]

        # Some toutatis distributions has an undocumented 7th line of 0's
        Table = numpy.fromfile(fin, dtype=numpy.float64, count=self.Np * 7 + 1)
        if len(Table) == self.Np * 7 + 1:
            self._data = Table[:-1].reshape(self.Np, 7)
        elif len(Table) == self.Np * 6 + 1:  # this is true in most cases
            self._data = Table[:-1].reshape(self.Np, 6)
        else:
            raise ValueError("Incorrect table dimensions found:", len(Table))

        # convert x,y from cm to m:
        self._data[:, 0] *= 1e-2
        self._data[:, 2] *= 1e-2

        self.mass = Table[-1]

    def keys(self):
        return self._columns[:]

    def __getitem__(self, key):
        # makes the class function as a dictionary
        # e.g. dst['x'] returns the x array..
        try:
            i = self._columns.index(key)
            return self._data[:, i]
        except:
            raise ValueError("Available keys: " + str(self._columns))

    def __setitem__(self, key, value):
        try:
            i = self._columns.index(key)
            self._data[:, i] = value
        except:
            raise ValueError("Available keys: " + str(self._columns))

    def save(self, filename, toutatis=False):
        """
        Save the distribution file
        so it can be read by TraceWin again

        :param filename: Name of file
        :param toutatis: Include 7th column of zeros

        Stolen from Ryoichi's func.py (with permission)
        """

        from struct import pack
        import numpy

        fout = open(filename, "wb")
        fout.write(pack("b", 125))
        fout.write(pack("b", 100))
        fout.write(pack("i", self.Np))
        fout.write(pack("d", self.Ib))
        fout.write(pack("d", self.freq))
        fout.write(pack("b", 125))

        data = self._data.copy()

        if toutatis and data.shape[1] == 6:
            data = numpy.append(data, numpy.zeros((len(data), 1)), 1)
        elif not toutatis and data.shape[1] == 7:
            data = data[:, :-1]

        # convert x,y from m to cm:
        data[:, 0] *= 1e2
        data[:, 2] *= 1e2

        if toutatis:
            data = data.reshape(self.Np * 7, 1)
        else:
            data = data.reshape(self.Np * 6, 1)

        for x in data:
            fout.write(pack("d", x))

        fout.write(pack("d", self.mass))
        fout.close()

    def subplot(self, index, x, y=None, nb=100, mask=None):
        """
        Create a subplot histogram similar to TraceWin.

        Example::
            import numpy as np
            from ess import TraceWin
            from matplotlib import pyplot as plt
            data=TraceWin.dst('part_dtl1.dst')
            m=np.where(data['E']>3.5)
            data.subplot(221,'x','xp',mask=m)
            data.subplot(222,'y','yp',mask=m)
            data.subplot(223,'phi','E',mask=m)
            data.subplot(224,'x','y',mask=m)
            plt.show()

        """
        from matplotlib.colors import LogNorm
        import matplotlib.pyplot as plt
        import numpy as np

        units = {
            "x": "mm",
            "y": "mm",
            "xp": "mrad",
            "yp": "mrad",
            "E": "MeV",
            "phi": "deg",
        }
        # get X and Y data
        dx = np.array(self[x])
        if mask != None:
            dx = dx[mask]
        if y != None:
            dy = np.array(self[y])
            if mask != None:
                dy = dy[mask]

        if x in ["x", "y", "xp", "yp"]:
            dx *= 1e3
        if y in ["x", "y", "xp", "yp"]:
            dy *= 1e3
        if x in ["phi"]:
            dx -= np.average(dx)
            dx *= 180 / np.pi
        if y in ["phi"]:
            dy -= np.average(dy)
            dy *= 180 / np.pi
        if x in ["E"] and max(dx) < 0.1:
            dx *= 1e3
            units["E"] = "keV"
        if y in ["E"] and max(dy) < 0.1:
            dy *= 1e3
            units["E"] = "keV"

        plt.subplot(index)
        if y != None:
            plt.hist2d(dx, dy, bins=nb, norm=LogNorm())
            plt.title("{} [{}] - {} [{}]".format(x, units[x], y, units[y]))
            hist, bin_edges = np.histogram(dx, bins=nb)
            b = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
            plt.plot(
                b,
                hist * 0.2 * (max(dy) - min(dy)) / max(hist) + min(dy),
                "k",
                lw=1.5,
                drawstyle="steps",
            )
            hist, bin_edges = np.histogram(dy, bins=nb)
            b = bin_edges[:-1] + 0.0 * (bin_edges[1] - bin_edges[0])
            plt.plot(
                hist * 0.2 * (max(dx) - min(dx)) / max(hist) + min(dx),
                b,
                "k",
                lw=1.5,
                drawstyle="steps",
            )
        else:
            # plot a simple 1D histogram..
            plt.hist(dx, bins=nb)
            plt.title("{} [{}]".format(x, units[x]))


class Plt:
    """
    Simple class to read in a
    TraceWin plot file

    Class afterwards hold the following
    dictionary items:
      - Ne (number of locations)
      - Np (number of particles)
      - Ib [A] (beam current)
      - freq [MHz]
      - mc2  [MeV]
      - Nelp [m] (locations)

    each plt[i], where i is element number, holds:
      - Zgen [cm] (location)
      - phase0 [deg] (ref phase)
      - wgen [MeV] (ref energy)
      - x [array, m]
      - xp [array, rad]
      - y [array, m]
      - yp [array, rad]
      - phi [array, rad]
      - E [array, MeV]
      - l [array] (is lost)

      Example::
        plt=ess.TraceWin.plt('calc/dtl1.plt')
        for i in [97,98]:
          data=plt[i]$
          if data:
            print(data['x'])
    """

    def __init__(self, filename):
        # easy storage..
        self.filename = filename
        # used to create dict behaviour..
        self._columns = ["x", "xp", "y", "yp", "phi", "E", "l"]
        # read in the file..
        self._readBinaryFile()

    def _readBinaryFile(self):
        # Thanks Emma!

        import numpy

        fin = open(self.filename, "r")

        # dummy, Np, Ib, freq, dummy
        Header_type = numpy.dtype(
            [
                ("dummy12", numpy.int16),
                ("Ne", numpy.int32),
                ("Np", numpy.int32),
                ("Ib", numpy.float64),
                ("freq", numpy.float64),
                ("mc2", numpy.float64),
            ]
        )
        SubHeader_type = numpy.dtype(
            [
                ("dummy12", numpy.int8),
                ("Nelp", numpy.int32),
                ("Zgen", numpy.float64),
                ("phase0", numpy.float64),
                ("wgen", numpy.float64),
            ]
        )

        Header = numpy.fromfile(fin, dtype=Header_type, count=1)
        self.Np = Header["Np"][0]
        self.Ne = Header["Ne"][0]
        self.Ib = Header["Ib"][0]
        self.freq = Header["freq"][0]
        self.mc2 = Header["mc2"][0]

        self._data = []
        self.Nelp = []

        i = 0
        while i < self.Ne:
            SubHeader = numpy.fromfile(fin, dtype=SubHeader_type, count=1)
            # unfinished files need this fix (simulation still running)
            if len(SubHeader["Nelp"]) == 0:
                break
            i = SubHeader["Nelp"][0]

            self.Nelp.append(i)

            Table = numpy.fromfile(fin, dtype=numpy.float32, count=self.Np * 7)
            Table = Table.reshape(self.Np, 7)
            data = {}
            for key in ["Zgen", "phase0", "wgen"]:
                data[key] = SubHeader[key][0]
            for j in range(7):
                c = self._columns[j]
                data[c] = Table[:, j]
                # convert x,y from cm to m
                if c in ["x", "y"]:
                    data[c] *= 1e-2
            self._data.append(data)

    def __getitem__(self, key):
        if key in self.Nelp:
            import numpy

            i = self.Nelp.index(key)

            ret = {}
            # some particles are lost, exclude those:
            lost_mask = self._data[i]["l"] == 0
            for key in self._data[i]:
                if isinstance(self._data[i][key], numpy.ndarray):
                    ret[key] = self._data[i][key][lost_mask]
                else:
                    ret[key] = self._data[i][key]
            return ret
        else:
            print("No data to plot at element", key)

    def calc_s(self):
        """
        Generates self.s which holds
        the position of each element
        in metres
        """
        import numpy

        self.s = []
        for i in self.Nelp:
            self.s.append(self[i]["Zgen"] / 100.0)
        self.s = numpy.array(self.s)

    def calc_avg(self):
        """
        Calculates averages of 6D coordinates at each
        element, such that e.g.
        self.avg["x"] gives average X at each location.

        Units: m, rad, MeV
        """
        import numpy

        self.avg = dict(x=[], xp=[], y=[], yp=[], E=[], phi=[])

        vals = self._columns[:-1]

        for i in self.Nelp:
            data = self[i]
            for v in vals:
                self.avg[v].append(numpy.average(data[v]))

    def calc_rel(self):
        """
        Calculates relativistic gamma/beta
        at each position, based on
        AVERAGE beam energy
        (NOT necessarily reference)
        """
        import numpy

        if not hasattr(self, "avg"):
            self.calc_avg()
        self.gamma = []
        self.beta = []
        for i, j in zip(self.Nelp, range(len(self.Nelp))):
            Eavg = self.avg["E"][j]
            self.gamma.append((self.mc2 + Eavg) / self.mc2)
            self.beta.append(numpy.sqrt(1.0 - 1.0 / self.gamma[-1] ** 2))
        self.gamma = numpy.array(self.gamma)
        self.beta = numpy.array(self.beta)

    def calc_minmax(self, pmin=5, pmax=95):
        """
        Calculates min/max values of beam coordinates
        in percentile, pmin is lower and pmax upper.

        Units: cm
        """
        import numpy

        self.min = dict(x=[], xp=[], y=[], yp=[], E=[])
        self.max = dict(x=[], xp=[], y=[], yp=[], E=[])

        for i in self.Nelp:
            data = self[i]
            for v in self.min.keys():
                self.min[v].append(numpy.percentile(data[v], pmin))
                self.max[v].append(numpy.percentile(data[v], pmax))

        for v in self.min.keys():
            self.min[v] = numpy.array(self.min[v])
            self.max[v] = numpy.array(self.max[v])

    def calc_sigma(self):
        """
        Calculates the sigma matrix

        Creates self.sigma such that self.sigma[i,j]
        returns the sigma matrix for value i,j.

        The numbering is:
        0: x
        1: xp
        2: y
        3: yp
        4: E
        5: phi
        """

        import numpy

        if not hasattr(self, "avg"):
            self.calc_avg()

        vals = self._columns[:-1]

        self.sigma = []
        for j in range(len(self.Nelp)):
            i = self.Nelp[j]
            data = self[i]

            self.sigma.append(
                [
                    [
                        numpy.mean(
                            (data[n] - self.avg[n][j]) * (data[m] - self.avg[m][j])
                        )
                        for n in vals
                    ]
                    for m in vals
                ]
            )

        self.sigma = numpy.array(self.sigma)

    def calc_std(self):
        """
        Calculates the beam sizes

        """

        import numpy

        if not hasattr(self, "sigma"):
            self.calc_sigma()

        vals = self._columns[:-1]

        self.std = {}

        for j in range(len(vals)):
            v = vals[j]
            self.std[v] = numpy.sqrt(self.sigma[:, j, j])

    def calc_twiss(self):
        """
        Calculates emittance, beta, alfa, gamma
        for each plane, x-xp, y-yp, and E-phi
        """

        import numpy

        if not hasattr(self, "sigma"):
            self.calc_sigma()
        if not hasattr(self, "gamma"):
            self.calc_rel()

        self.twiss_eps = []
        for j in range(len(self.Nelp)):
            self.twiss_eps.append(
                [
                    numpy.sqrt(numpy.linalg.det(self.sigma[j][i : i + 2][:, i : i + 2]))
                    for i in (0, 2, 4)
                ]
            )
        self.twiss_eps = numpy.array(self.twiss_eps)

        # Calculate normalized emittance:
        # TODO: this is NOT correct normalization for longitudinal
        self.twiss_eps_normed = self.twiss_eps.copy()
        for i in range(3):
            self.twiss_eps_normed[:, i] *= self.gamma * self.beta

        # Calculate beta:
        # This is a factor 10 different from what TraceWin plots
        self.twiss_beta = [
            [self.sigma[j][i][i] / self.twiss_eps[j, i // 2] for i in (0, 2, 4)]
            for j in range(len(self.Nelp))
        ]
        self.twiss_beta = numpy.array(self.twiss_beta)

        # Calculate alpha:
        self.twiss_alpha = [
            [-self.sigma[j][i][i + 1] / self.twiss_eps[j, i // 2] for i in (0, 2, 4)]
            for j in range(len(self.Nelp))
        ]
        self.twiss_alpha = numpy.array(self.twiss_alpha)

    def get_dst(self, index):
        """
        Returns the dst corresponding to the given index
        """
        import numpy

        dset = self[index]

        _dst = Dst()
        _dst.freq = self.freq
        _dst.Ib = self.Ib * 1000
        _dst.Np = len(dset["x"])
        _dst.mass = self.mc2
        _dst._data = numpy.array(
            [dset["x"], dset["xp"], dset["y"], dset["yp"], dset["phi"], dset["E"]]
        ).transpose()
        return _dst

    def save_dst(self, index, filename):
        """
        Saves the dst at the specified index to file

        Returns the same dst object.
        """
        _dst = self.get_dst(index)
        _dst.save(filename)
        return _dst


class DensityFile:
    """
    Simple class to read a TraceWin density file
    into a pythonized object
    """

    def __init__(self, filename, envelope=None):
        import numpy
        import sys

        self.filename = filename
        self.fin = open(self.filename, "r")

        if envelope is None:  # try to guess
            if filename.split("/")[-1].split(".")[0] == "Density_Env":
                self.envelope = True
            else:
                self.envelope = False
        else:
            self.envelope = envelope

        # currently unknown:
        self.version = 0

        # first we simply count how many elements we have:
        counter = 0
        while True:
            try:
                self._skipAndCount()
                counter += 1
            except IndexError:  # EOF reached..
                break
        if sys.flags.debug:
            print("Number of steps found:", counter)
        self.fin.seek(0)

        # set up the arrays..
        self.i = 0
        # z position [m] :
        self.z = numpy.zeros(counter)
        # element index number
        self.nelp = numpy.zeros(counter)
        # current [mA] :
        self.ib = numpy.zeros(counter)
        # number of lost particles:
        self.Np = numpy.zeros(counter)

        self.Xouv = numpy.zeros(counter)
        self.Youv = numpy.zeros(counter)

        if self.version >= 9:
            self.dXouv = numpy.zeros(counter)
            self.dYouv = numpy.zeros(counter)

        self.moy = numpy.zeros((counter, 7))
        self.moy2 = numpy.zeros((counter, 7))

        self._max = numpy.zeros((counter, 7))
        self._min = numpy.zeros((counter, 7))

        if self.version >= 10:
            self.maxR = numpy.zeros((counter, 7))
            self.minR = numpy.zeros((counter, 7))

        if self.version >= 5:
            self.rms_size = numpy.zeros((counter, 7))
            self.rms_size2 = numpy.zeros((counter, 7))

        if self.version >= 6:
            self.min_pos_moy = numpy.zeros((counter, 7))
            self.max_pos_moy = numpy.zeros((counter, 7))

        if self.version >= 7:
            self.rms_emit = numpy.zeros((counter, 3))
            self.rms_emit2 = numpy.zeros((counter, 3))

        if self.version >= 8:
            self.energy_accept = numpy.zeros(counter)
            self.phase_ouv_pos = numpy.zeros(counter)
            self.phase_ouv_neg = numpy.zeros(counter)

        self.lost = numpy.zeros((counter, self.Nrun))
        self.powlost = numpy.zeros((counter, self.Nrun))

        self.lost2 = numpy.zeros(counter)
        self.Minlost = numpy.zeros(counter)
        self.Maxlost = numpy.zeros(counter)

        self.powlost2 = numpy.zeros(counter)
        self.Minpowlost = numpy.zeros(counter)
        self.Maxpowlost = numpy.zeros(counter)

        while self.i < counter:
            self._getFullContent()
            self.i += 1
            if sys.flags.debug and self.i % 100 == 0:
                print("Read status", self.i)

    def _getHeader(self):
        import numpy

        # header..
        version = numpy.fromfile(self.fin, dtype=numpy.int16, count=1)[0]
        year = numpy.fromfile(self.fin, dtype=numpy.int16, count=1)[0]

        # in case we did not read all data, this will detect our mistake:
        shift = 0
        while year != 2011 or version not in [8, 9, 10, 11, 12]:
            shift += 1
            version = year
            year = numpy.fromfile(self.fin, dtype=numpy.int16, count=1)[0]
        if shift:
            print(year, version)
            raise ValueError("ERROR, shifted " + str(shift * 2) + " bytes")

        self.vlong = numpy.fromfile(self.fin, dtype=numpy.int16, count=1)[0]
        self.Nrun = numpy.fromfile(self.fin, dtype=numpy.int32, count=1)[0]

        self.version = version
        self.year = year

    def _skipAndCount(self):
        import numpy

        self._getHeader()

        if self.envelope:
            if self.version == 8:
                numpy.fromfile(self.fin, dtype=numpy.int16, count=292 // 2)
            elif self.version == 9:
                numpy.fromfile(self.fin, dtype=numpy.int16, count=300 // 2)
            elif self.version == 10:
                numpy.fromfile(self.fin, dtype=numpy.int16, count=356 // 2)
            else:
                raise TypeError("It is not possible to read this format..")
        elif self.Nrun > 1:
            # WARN not 100% sure if this is correct..
            if self.version <= 9:
                numpy.fromfile(
                    self.fin, dtype=numpy.int16, count=((5588 + self.Nrun * 12) // 2)
                )
            elif self.version == 10:
                numpy.fromfile(
                    self.fin, dtype=numpy.int16, count=((20796 + self.Nrun * 12) // 2)
                )
            else:
                raise TypeError("It is not possible to read this format..")
        elif self.version == 8:
            numpy.fromfile(self.fin, dtype=numpy.int16, count=12344 // 2)
        elif self.version == 9:
            numpy.fromfile(self.fin, dtype=numpy.int16, count=12352 // 2)
        elif self.version == 10:
            numpy.fromfile(self.fin, dtype=numpy.int16, count=12408 // 2)
        else:
            raise TypeError("It is not possible to read this format..")

    def _get_7dim_array(array):
        """
        Unused?
        """
        return dict(
            x=array[0],
            y=array[1],
            phase=array[2],
            energy=array[3],
            r=array[4],
            z=array[5],
            dpp=array[6],
        )

    def _getFullContent(self):

        import numpy

        # self._getHeader()
        # no need to read the header again:
        # (though only if we are SURE about content!)
        numpy.fromfile(self.fin, dtype=numpy.int16, count=5)

        self.nelp[self.i] = numpy.fromfile(self.fin, dtype=numpy.int32, count=1)[0]
        self.ib[self.i] = numpy.fromfile(self.fin, dtype=numpy.float32, count=1)[0]
        self.z[self.i] = numpy.fromfile(self.fin, dtype=numpy.float32, count=1)[0]
        # Aperture
        self.Xouv[self.i] = numpy.fromfile(self.fin, dtype=numpy.float32, count=1)[0]
        self.Youv[self.i] = numpy.fromfile(self.fin, dtype=numpy.float32, count=1)[0]
        if self.version >= 9:
            dXouv = numpy.fromfile(self.fin, dtype=numpy.float32, count=1)[0]
            dYouv = numpy.fromfile(self.fin, dtype=numpy.float32, count=1)[0]
        step = numpy.fromfile(self.fin, dtype=numpy.int32, count=1)[0]

        n = 7  # x [m], y[m], Phase [deg], Energy [MeV], R[m], Z[m], dp/p

        self.moy[self.i] = numpy.fromfile(self.fin, dtype=numpy.float32, count=n)[:]
        self.moy2[self.i] = numpy.fromfile(self.fin, dtype=numpy.float32, count=n)[:]

        self._max[self.i] = numpy.fromfile(self.fin, dtype=numpy.float32, count=n)[:]
        self._min[self.i] = numpy.fromfile(self.fin, dtype=numpy.float32, count=n)[:]

        if self.version >= 10:
            self.maxR[self.i] = numpy.fromfile(self.fin, dtype=numpy.float32, count=n)[
                :
            ]
            self.minR[self.i] = numpy.fromfile(self.fin, dtype=numpy.float32, count=n)[
                :
            ]

        if self.version >= 5:
            self.rms_size[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float32, count=n
            )
            self.rms_size2[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float32, count=n
            )
        if self.version >= 6:
            self.min_pos_moy[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float32, count=n
            )
            self.max_pos_moy[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float32, count=n
            )
        if self.version >= 7:
            self.rms_emit[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float32, count=3
            )[:]
            self.rms_emit2[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float32, count=3
            )[:]
        if self.version >= 8:
            self.energy_accept[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float32, count=1
            )
            self.phase_ouv_pos[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float32, count=1
            )
            self.phase_ouv_neg[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float32, count=1
            )

        self.Np[self.i] = numpy.fromfile(self.fin, dtype=numpy.int64, count=1)[0]

        if self.Np[self.i]:
            for i in range(self.Nrun):
                self.lost[self.i, i] = numpy.fromfile(
                    self.fin, dtype=numpy.int64, count=1
                )[0]
                self.powlost[self.i, i] = numpy.fromfile(
                    self.fin, dtype=numpy.float32, count=1
                )[0]
            self.lost2[self.i] = numpy.fromfile(self.fin, dtype=numpy.int64, count=1)[0]
            self.Minlost[self.i] = numpy.fromfile(self.fin, dtype=numpy.int64, count=1)[
                0
            ]
            self.Maxlost[self.i] = numpy.fromfile(self.fin, dtype=numpy.int64, count=1)[
                0
            ]
            self.powlost2[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float64, count=1
            )[0]
            self.Minpowlost[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float32, count=1
            )[0]
            self.Maxpowlost[self.i] = numpy.fromfile(
                self.fin, dtype=numpy.float32, count=1
            )[0]

            if self.vlong == 1:
                tab = numpy.fromfile(self.fin, dtype=numpy.uint64, count=n * step)
            else:
                tab = numpy.fromfile(self.fin, dtype=numpy.uint32, count=n * step)

            if self.ib[self.i] > 0:
                tabp = numpy.fromfile(self.fin, dtype=numpy.uint32, count=3 * step)

    def _avg_merge(self, other, param):
        """
        returns the average of the parameter
        weighted by how many Nruns in self and other object

        This allows for different lengths of the two arrays..
        """
        mine = getattr(self, param)
        new = getattr(other, param)
        if len(mine) > len(new):
            ret = mine.copy()
            ret[: len(new)] = (mine[: len(new)] * self.Nrun + new * other.Nrun) / (
                self.Nrun + other.Nrun
            )
        elif len(mine) < len(new):
            ret = new.copy()
            ret[: len(mine)] = (mine * self.Nrun + new[: len(mine)] * other.Nrun) / (
                self.Nrun + other.Nrun
            )
        else:
            ret = (mine * self.Nrun + new * other.Nrun) / (self.Nrun + other.Nrun)
        return ret

    def _sum_merge(self, other, param):
        """
        returns the sum of the parameter

        This allows for different lengths of the two arrays..
        """
        mine = getattr(self, param)
        new = getattr(other, param)
        if len(mine) > len(new):
            ret = mine.copy()
            ret[: len(new)] += new
        elif len(mine) < len(new):
            ret = new.copy()
            ret[: len(mine)] += mine
        else:
            ret = mine + new
        return ret

    def _concatenate_merge(self, other, param):
        """
        returns the concatenation of the two matrices

        This allows for different lengths of the two arrays/matrices..
        """
        import numpy

        mine = getattr(self, param)
        new = getattr(other, param)
        ret = numpy.zeros((max([len(mine), len(new)]), len(mine[0]) + len(new[0])))
        ret[: len(mine), : len(mine[0])] = mine
        ret[: len(new), len(mine[0]) :] = new
        return ret

    def _fun_merge(self, other, function, param):
        """
        returns the function applied on the parameter

        This allows for different lengths of the two arrays..
        """
        mine = getattr(self, param)
        new = getattr(other, param)
        if len(mine) > len(new):
            ret = mine.copy()
            ret[: len(new)] = function(mine[: len(new)], new)
        elif len(mine) < len(new):
            ret = new.copy()
            ret[: len(mine)] = function(mine, new[: len(mine)])
        else:
            ret = function(mine, new)
        return ret

    def merge(self, objects):
        """
        Merge with list of objects
        """
        import numpy

        if not isinstance(objects, list):
            raise TypeError("You tried to merge a non-list")

        # for now we only allow objects with same version..
        for o in objects:
            if self.version != o.version:
                raise ValueError("Cannot merge files with differing version")

        # merge info..
        for o in objects:
            if len(self.ib) < len(o.ib):
                raise ValueError("Sorry, not implemented yet. Complain to Yngve")

            self.ib = self._avg_merge(o, "ib")

            # this looks strange to me, but it is what TraceWin does..
            self.moy = self._sum_merge(o, "moy")
            self.moy2 = self._sum_merge(o, "moy")

            self._max = self._fun_merge(o, numpy.maximum, "_max")
            self._min = self._fun_merge(o, numpy.minimum, "_min")

            if self.version >= 5:
                # this looks strange to me, but it is what TraceWin does..
                self.rms_size = self._sum_merge(o, "rms_size")
                self.rms_size2 = self._sum_merge(o, "rms_size2")

            if self.version >= 6:
                self.max_pos_moy = self._fun_merge(o, numpy.maximum, "max_pos_moy")
                self.min_pos_moy = self._fun_merge(o, numpy.minimum, "min_pos_moy")

            if self.version >= 7:
                # this looks strange to me, but it is what TraceWin does..
                self.rms_emit = self._sum_merge(o, "rms_emit")
                self.rms_emit2 = self._sum_merge(o, "rms_emit2")

            if self.version >= 8:
                # Warning: TraceWin does NOT merge these data in any way
                self.energy_accept = self._avg_merge(o, "energy_accept")
                self.phase_ouv_pos = self._avg_merge(o, "phase_ouv_pos")
                self.phase_ouv_neg = self._avg_merge(o, "phase_ouv_neg")

            # Note, we don't get into the problem of differing table sizes
            # particles are lost, because we have written zeroes for
            # the rest of the tables

            self.lost = self._concatenate_merge(o, "lost")
            self.powlost = self._concatenate_merge(o, "powlost")

            self.lost2 = self._sum_merge(o, "lost2")
            self.powlost2 = self._sum_merge(o, "powlost2")

            self.Minlost = self._fun_merge(o, numpy.minimum, "Minlost")
            self.Maxlost = self._fun_merge(o, numpy.maximum, "Maxlost")
            self.Minpowlost = self._fun_merge(o, numpy.minimum, "Minpowlost")
            self.Maxpowlost = self._fun_merge(o, numpy.maximum, "Maxpowlost")

            # Note: We are ignoring tab/tabp data...

            # merge final info (make sure to do this last!)
            self.Np = self._sum_merge(o, "Np")
            self.Nrun += o.Nrun

    def savetohdf(self, filename="Density.h5", group="TraceWin", force=False):
        """
        Saves data to HDF5
        """
        import h5py
        import sys

        fout = h5py.File(filename, "a")
        if group in fout:
            if force:
                del fout[group]
            else:
                if sys.flags.debug:
                    print("Group {} already exist in {}".format(group, filename))
                return

        group = fout.create_group(group)

        # header attributes..
        group.attrs["version"] = self.version
        group.attrs["year"] = self.year
        group.attrs["Nrun"] = self.Nrun
        group.attrs["vlong"] = self.vlong

        length = len(self.z)

        partran = sum(self.Np) > 0

        # one number per location
        arrays = ["z", "nelp", "ib", "Np", "Xouv", "Youv"]
        array_units = ["m", "", "mA", "", "m", "m"]
        if self.version >= 8:
            arrays += ["energy_accept", "phase_ouv_pos", "phase_ouv_neg"]
            array_units += ["eV", "deg", "deg"]
        if partran:
            arrays += [
                "lost2",
                "Minlost",
                "Maxlost",
                "powlost2",
                "Minpowlost",
                "Maxpowlost",
            ]
            array_units += ["", "", "", "W*w", "W", "W"]

        # 7 numbers per location..
        coordinates = ["moy", "moy2", "_max", "_min"]
        coordinate_units = ["m", "m*m", "m", "m"]
        if self.version >= 5 and partran:
            coordinates += ["rms_size", "rms_size2"]
            coordinate_units += ["m", "m*m"]
        if self.version >= 6 and partran:
            coordinates += ["min_pos_moy", "max_pos_moy"]
            coordinate_units += ["m", "m"]

        for val, unit in zip(arrays, array_units):
            data_set = group.create_dataset(val, (length,), dtype="f")
            data_set[...] = getattr(self, val)
            if unit:
                data_set.attrs["unit"] = unit

        for val, unit in zip(coordinates, coordinate_units):
            data_set = group.create_dataset(val, (length, 7), dtype="f")
            data_set[...] = getattr(self, val)
            if unit:
                data_set.attrs["unit"] = unit

        if self.version >= 7 and partran:
            # 3 numbers per location..
            emit_data = ["rms_emit", "rms_emit2"]
            emit_units = ["m*rad", "m*m*rad*rad"]
            for val, unit in zip(emit_data, emit_units):
                data_set = group.create_dataset(val, (length, 3), dtype="f")
                data_set[...] = getattr(self, val)
                if unit:
                    data_set.attrs["unit"] = unit
        if partran:
            # 1 numbers per location and per run..
            data = ["lost", "powlost"]
            units = ["", "W"]
            for val, unit in zip(data, units):
                data_set = group.create_dataset(val, (length, self.Nrun), dtype="f")
                data_set[...] = getattr(self, val)
                if unit:
                    data_set.attrs["unit"] = unit

        fout.close()


class RemoteDataMerger:
    def __init__(self, base="."):
        self._base = base
        self._files = []

    def add_file(self, filepath):
        import os

        if os.path.exists(filepath):
            fname = filepath
        else:
            fullpath = os.path.join(self._base, filepath)
            if os.path.exists(fullpath):
                fname = fullpath
            else:
                raise ValueError("Could not find file " + filepath)
        if fname not in self._files:
            self._files.append(fname)

    def generate_partran_out(self, filename=None):
        """
        Creates a string to be written to file
        each line is a list.

        If filename is given, writes directly to output file.

        """

        import numpy as np

        h1 = []
        h2 = []

        d1 = []
        d2 = []
        d3 = []

        if self._files:
            for f in self._files:
                string = open(f, "r").read()
                split = string.split("$$$")
                if split[9] != "Data_Error":
                    raise ValueError("Magic problem, please complain to Yngve")

                thisdata = split[10].strip().split("\n")

                if not h1:
                    h1 = [thisdata[0] + " (std in paranthesis)"]
                    h2 = thisdata[2:10]
                d1.append(thisdata[1].split())
                d2.append(thisdata[10])
                d3.append(thisdata[11])

            # fix d1:
            for i in range(len(d1)):
                for j in range(len(d1[0])):
                    d1[i][j] = float(d1[i][j])
            d1 = np.array(d1)
            means = d1.mean(axis=0)
            stds = d1.std(axis=0)
            d1 = []
            for i in range(len(stds)):
                if stds[i] / means[i] < 1e-10:
                    stds[i] = 0.0
            for i in range(len(stds)):
                # some small std are removed..
                if stds[i] / means[i] > 1e-8:
                    d1.append("%f(%f)" % (means[i], stds[i]))
                else:  # error is 0
                    d1.append(str(means[i]))
            d1 = [" ".join(d1)]

            # create data:
            data = h1 + d1 + h2 + d2 + d3

            if filename:
                open(filename, "w").write("\n".join(data))

            return data


class Partran(dict):
    """
    Read partran1.out files..
    """

    def __init__(self, filename):
        self.filename = filename
        self._readAsciiFile()

    def _readAsciiFile(self):

        import numpy

        stream = open(self.filename, "r")
        for i in range(10):
            line = stream.readline()
            if line.strip()[0] == "#":
                break
        self.columns = ["NUM"] + line.split()[1:]
        self.data = numpy.loadtxt(stream)

        self._dict = {}
        for i in range(len(self.columns)):
            self[self.columns[i]] = self.data[:, i]


class FieldMap:
    """
    Class to read in the field map structures

    WARNING: Work in progress!!
    """

    def __init__(self, filename):
        self._filename = filename
        self._load_data(filename)

    def _load_data(self, filename):
        import os
        import numpy

        if not os.path.isfile(filename):
            raise ValueError("Cannot find file {}".format(filename))
        fin = open(filename, "r")
        l = fin.readline().split()
        self.start = []
        self.end = []
        numindexes = []
        while len(l) > 1:
            numindexes.append(int(l[0]) + 1)
            if len(l) == 2:
                self.start.append(0.0)
                self.end.append(float(l[1]))
            else:
                self.start.append(float(l[1]))
                self.end.append(float(l[2]))
            l = fin.readline().split()

        self.z = numpy.arange(
            self.start[0], self.end[0], (self.end[0] - self.start[0]) / numindexes[0]
        )
        if len(self.start) > 1:
            self.y = numpy.arange(
                self.start[1],
                self.end[1],
                (self.end[1] - self.start[1]) / numindexes[1],
            )
        if len(self.start) > 2:
            self.x = numpy.arange(
                self.start[2],
                self.end[2],
                (self.end[2] - self.start[2]) / (numindexes[2]),
            )

        self.norm = float(l[0])
        self.map = numpy.loadtxt(fin).reshape(numindexes)

    def savemap(self, filename):
        fout = open(filename, "w")
        for n, s in zip(self.map.shape, self.size):
            fout.write("{} {}\n".format(n - 1, s))
        fout.write("{}\n".format(self.norm))
        totmapshape = 1
        for i in self.map.shape:
            totmapshape *= i
        data = self.map.reshape(totmapshape)
        for j in data:
            fout.write("{}\n".format(j))
